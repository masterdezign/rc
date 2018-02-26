-- |
-- = Nonlinear transient computing
--
-- This module was developed as a part of author's PhD project:
-- https://www.researchgate.net/project/Theory-and-Modeling-of-Complex-Nonlinear-Delay-Dynamics-Applied-to-Neuromorphic-Computing
--

{-# LANGUAGE BangPatterns #-}

module RC.NTC
  ( new
  , learn
  , learnClassifier
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
import qualified Data.List as List
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import           System.IO.Unsafe ( unsafePerformIO )
import qualified Learning
import qualified Numeric.DDE as DDE
import qualified Numeric.DDE.Model as DDEModel

import qualified RC.Helpers as H

-- | Reservoir abstraction.
--
-- Reservoir (a recurrent neural network) is an essential
-- component in the NTC framework.
--
-- In this project, we exploit an established analogy between
-- spatially extended and delay systems ^1-3. That makes possible
-- to employ DDEs as a reservoir substrate. The choice of substrate
-- does not affect the `Reservoir` definition below.
--
-- ^1 Arecchi, F. T., et al. “Two-Dimensional Representation of
--    a Delayed Dynamical System.” Physical Review A, vol. 45, no. 7,
--    Jan. 1992, doi:10.1103/physreva.45.r4225.
-- ^2 Appeltant, L., et al. “Information Processing Using a Single
--    Dynamical Node as Complex System.” Nature Communications, vol. 2,
--    2011, p. 468., doi:10.1038/ncomms1476.
-- ^3 Virtual Chimera States for Delayed-Feedback Systems - Laurent Larger,
--    Bogdan Penkovsky, Yuri Maistrenko. Physical Review Letters - 08 / 2013
--
newtype Reservoir = Reservoir { _transform :: Matrix Double -> Matrix Double }

-- | Customizable NTC parameters
data NTCParameters = Par
  { _preprocess :: Matrix Double -> Matrix Double
    -- ^ Modify data before masking (e.g. compression)
  , _inputWeightsRange :: (Double, Double)  -- ^ Input weights (mask) range
  , _inputWeightsGenerator :: StdGen -> (Int, Int) -> (Double, Double) -> Matrix Double
  , _postprocess :: Matrix Double -> Matrix Double
  -- ^ Modify data before training or prediction (e.g. add biases)
  , _reservoirModel :: DDEModel.Par
  }

-- | NTC network structure
data NTC = NTC
  { _inputWeights :: Matrix Double
  , _reservoir :: Reservoir
  , _outputWeights :: Maybe (Matrix Double)
  -- ^ Trainable part of NTC
  , _par :: NTCParameters
  -- ^ Number of hidden nodes
  }

-- | An untrained NTC network
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

-- | Substrate-specific low-level reservoir implementation
genReservoir :: DDEModel.Par -> Reservoir
genReservoir par@DDEModel.RC {
    DDEModel._filt = DDEModel.BandpassFiltering { DDEModel._tau = tau }
  } = Reservoir _r
  where
    _r sample = unflatten' nodes response
      where
        oversampling = 1 :: Int  -- No oversampling
        detuning = 1.0 :: Double  -- Delay detuning factor, 1 = no detuning
        nodes = rows sample
        delaySamples = round $ detuning * fromIntegral (oversampling * nodes)

        -- Matrix to timetrace
        trace1 = flatten' sample

        -- Duplicate the last element (DDE.integHeun2_2D consumes one extra input)
        trace = trace1 V.++ V.singleton (V.last trace1)

        -- Empirically chosen integration time step:
        -- twice faster than the system response time tau
        hStep = tau / 2

        !(_, !response) = DDE.integHeun2_2D delaySamples hStep (DDEModel.rhs par) (DDE.Input trace)

genReservoir _ = error "Unsupported DDE model"

-- | Nonlinear transformation performed by an NTC network
forwardPass :: NTC  -- ^ NTC network
            -> Matrix Double  -- ^ Input information
            -> Matrix Double
forwardPass NTC { _par = Par { _preprocess = prep, _postprocess = post }
                , _inputWeights = iw
                , _reservoir = Reservoir res
                } !sample =
  let pipeline = post. res. (iw <>). prep
  in pipeline sample

-- | NTC training: learn the readout weights offline
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

_concatForwardPass :: NTC
                   -> Int
                   -- ^ Number of samples in the list (for
                   -- memory allocation)
                   -> [Matrix Double]
                   -- ^ List of samples
                   -> Matrix Double
_concatForwardPass ntc m (state0_:samples') = unsafePerformIO $ do
      -- Detect the postprocessing output dimension `nodes`.
      -- Alternatively, use Static shapes from hmatrix.
      let state0 = forwardPass ntc state0_
          nodes = rows state0
          state0' = flatten' state0

      -- Allocate memory for a mutable vector
      v <- VM.new (nodes * m)

      -- Copy the first computed state0'
      copy state0' v 0 0 nodes

      let foldA _ [] = return ()
          foldA f ((y, i):ys) = do
            let v0 = f y
            copy v0 v 0 (i * nodes) ((i + 1) * nodes)
            foldA f ys

      -- NB: Does compiler know (flatten'. unflatten') == id?
      -- If not, consider refactoring genReservoir and forwardPass
      foldA (flatten'. forwardPass ntc) $ zip samples' [1..m - 1]

      processed' <- V.unsafeFreeze v
      let processed = unflatten' nodes processed'
      return processed
  where
    copy !v0 !v !i0 !i !k
      | i < k = do
        VM.unsafeWrite v i (v0 V.! i0)
        copy v0 v (i0 + 1) (i + 1) k
      | otherwise = return ()

flatten' :: Matrix Double -> Vector Double
flatten' = flatten. tr

unflatten' :: Int -> Vector Double -> Matrix Double
unflatten' nodes = tr. reshape nodes

-- | NTC training specific to classification task.
-- The readout weights are learned offline.
--
-- Alternatively, one could use `learn` function. However,
-- to make sure the training samples do not mix in the reservoir RNN,
-- a significant padding of zeros would be needed. Although that
-- way directly corresponds to a physical experiment, that would be not
-- memory-efficient on a computer.
learnClassifier
  :: NTC
  -- ^ NTC network
  -> [Int]
  -- ^ Target labels
  -> Int
  -- ^ Number of inputs (for memory allocation)
  -> [Matrix Double]
  -- ^ Training inputs
  -> Either String (Learning.Classifier Int)
learnClassifier ntc labels m samples =
  case Learning.learn' state teacher' of
    Nothing -> Left "Cannot create a readout matrix"
    Just w -> let clf = Learning.winnerTakesAll w klasses. forwardPass ntc
           in Right (Learning.Classifier clf)
  where
    klasses = fromList. List.sort. List.nub $ labels
    klassesNo = V.length klasses
    teacher' = concatHor. map (flip (Learning.teacher klassesNo) 1) $ labels

    -- Alternatively, a streaming interface might be a solution
    -- https://hackage.haskell.org/package/pipes-4.3.7/docs/Pipes-Tutorial.html
    state = _concatForwardPass ntc m samples

-- | Horizontal matrix concatenation, alternative to fromBlocks
concatHor :: Element a => [Matrix a] -> Matrix a
concatHor ms@(m:_) = foldr (|||) (zeroCols (rows m)) ms
-- concatHor = concatMapHor id
{-# SPECIALIZE concatHor :: [Matrix Double] -> Matrix Double #-}

concatMapHor
  :: Element b =>
     (Matrix a -> Matrix b) -> [Matrix a] -> Matrix b
concatMapHor f ms@(m:_) = foldr (\a b -> (f a) ||| b) (zeroCols (rows m)) ms
{-# SPECIALIZE concatMapHor :: (Matrix Double -> Matrix Double) -> [Matrix Double] -> Matrix Double #-}

-- | Matrix with zero elements
zeroCols :: V.Storable a => Int -> Matrix a
zeroCols rows' = (rows'><0) []
{-# SPECIALIZE zeroCols :: Int -> Matrix Double #-}

-- | Run prediction using a "clean" (uninitialized) reservoir and then
-- forget the reservoir's state.
-- This can be used for both forecasting and classification tasks.
predict :: NTC  -- ^ Trained network
        -> Int  -- ^ Washout (forget) points
        -> Matrix Double
        -- ^ Input matrix where measurements are columns and features are rows
        -> Either String (Matrix Double)
        -- ^ Either error string or predicted output
predict ntc@NTC { _outputWeights = ow
                } forgetPts inp =
  case ow of
    Nothing -> Left "Please train the NTC first"
    Just w -> let y = forwardPass ntc inp
                  y2 = y ?? (All, Drop forgetPts)
                  prediction = w <> y2
              in Right prediction
