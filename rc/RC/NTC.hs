-- = Nonlinear transient computing
--
-- This module was developed as a part of author's PhD project:
-- https://www.researchgate.net/project/Theory-and-Modeling-of-Complex-Nonlinear-Delay-Dynamics-Applied-to-Neuromorphic-Computing
--

{-# LANGUAGE BangPatterns #-}

module RC.NTC
  ( new
  , learn
  , predict
  , par0
  , NTCParameters (..)
  , DDEModel.Par (..)
  , DDEModel.BandpassFiltering (..)
  ) where

import           Numeric.LinearAlgebra
import           System.Random ( StdGen
                               , split
                               )
import qualified Data.Vector.Storable as V
import qualified Learning
import qualified Numeric.DDE as DDE
import qualified Numeric.DDE.Model as DDEModel

import qualified RC.Helpers as H

-- | DDE reservoir abstraction
newtype Reservoir = Reservoir { _transform :: Matrix Double -> Matrix Double }

data NTCParameters = Par
  { _preprocess :: Matrix Double -> Matrix Double
    -- ^ Modify data before masking (e.g. compression)
  , _inputWeightsRange :: (Double, Double)  -- ^ Input weights (mask) range
  , _inputWeightsGenerator :: StdGen -> (Int, Int) -> (Double, Double) -> Matrix Double
  , _postprocess :: Matrix Double -> Matrix Double
  -- ^ Modify data before training or prediction (e.g. add biases)
  , _reservoirModel :: DDEModel.Par
  }

data NTC = NTC
  { _inputWeights :: Matrix Double
  , _reservoir :: Reservoir
  , _outputWeights :: Maybe (Matrix Double)
  -- ^ Trainable part of NTC
  , _par :: NTCParameters
  }

new
  :: StdGen
  -> NTCParameters
  -> (Int, Int, Int)
  -- ^ Input dimension, network nodes, and output dimension
  -> NTC
new g par (ind, nodes, out) =
  let iwgen = _inputWeightsGenerator par
      iw = iwgen g (nodes, ind) (_inputWeightsRange par)
      ntc = NTC { _inputWeights = iw
                , _reservoir = genReservoir (_reservoirModel par)
                , _outputWeights = Nothing
                , _par = par
                }
  in ntc

-- | Default NTC parameters
par0 :: NTCParameters
par0 = Par
  { _preprocess = id
  , _inputWeightsGenerator = H.randMatrix
  , _postprocess = H.addBiases  -- Usually `id` will work
  , _inputWeightsRange = undefined  -- To be manually set, e.g. (-1, 1)
  , _reservoirModel = DDEModel.RC { DDEModel._filt = filt'
                                  , DDEModel._rho = 3.25
                                  , DDEModel._fnl = H.hsigmoid (1.09375, 1.5, 0.0)
                                  }
  }
  where
    filt' = DDEModel.BandpassFiltering {
              DDEModel._tau = 7.8125e-3
            , DDEModel._theta = recip 0.34375
            }

genReservoir :: DDEModel.Par -> Reservoir
genReservoir par = Reservoir _r
  where
    _r sample = unflatten response
      where
        flatten' = flatten. tr
        unflatten = tr. reshape nodes

        oversampling = 1 :: Int  -- No oversampling
        detuning = 1.0 :: Double  -- Delay detuning factor, 1 = no detuning
        nodes = rows sample
        delaySamples = round $ detuning * fromIntegral (oversampling * nodes)

        -- Matrix to timetrace
        trace1 = flatten' sample

        -- Duplicate the last element (DDE.integHeun2_2D consumes one extra input)
        trace = trace1 V.++ V.singleton (V.last trace1)

        tau = DDEModel._tau $ DDEModel._filt par
        hStep = tau / 2

        !(_, !response) = DDE.integHeun2_2D delaySamples hStep (DDEModel.rhs par) (DDE.Input trace)

forwardPass :: NTC -> Matrix Double -> Matrix Double
forwardPass NTC { _par = Par { _preprocess = prep, _postprocess = post }
                , _inputWeights = iw
                , _reservoir = Reservoir res
                } sample =
  let pipeline = post. res. (iw <>). prep
  in pipeline sample

-- | Offline NTC training
learn
  :: NTC
  -> Int
  -- ^ Discard the first N points
  -> Matrix Double
  -- ^ Input matrix of features rows and observations columns
  -> Matrix Double
  -- ^ Desired output matrix of observations columns
  -> Either String NTC
learn ntc forgetPts inp out = ntc'
  where
    state' = (forwardPass ntc inp) ?? (All, Drop forgetPts)
    teacher' = out ?? (All, Drop forgetPts)
    ntc' = case Learning.learn' state' teacher' of
      Nothing -> Left "Cannot create a readout matrix"
      w -> Right $ ntc { _outputWeights = w }

predict :: NTC
        -> Int
        -> Matrix Double
        -> Either String (Matrix Double)
predict ntc@NTC { _outputWeights = ow
                } forgetPts inp =
  case ow of
    Nothing -> Left "Please train the NTC first"
    Just w -> let y = forwardPass ntc inp
                  y2 = y ?? (All, Drop forgetPts)
                  prediction = w <> y2
              in Right prediction
