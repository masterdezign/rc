{- |

= Jaeger's echo state network (ESN)

Echo state networks (ESN) provide an architecture and supervised
learning principle for recurrent neural networks (RNNs) ^1.
The code below is intended for research purposes.

^1 http://www.scholarpedia.org/article/Echo_state_network

-}

module RC.ESN where

import           Numeric.LinearAlgebra
import           System.Random ( StdGen
                               , split
                               )
import           Data.Maybe ( fromMaybe
                            , fromJust
                            )
import qualified Data.Vector.Storable as V
import qualified Learning

import qualified RC.Helpers as H

data Reservoir = Reservoir
  { _activation :: Double -> Double
  , _weights :: Matrix Double
  }

data ESN = ESN
  { _inputWeights :: Matrix Double
  , _reservoir :: Reservoir
  , _feedbackWeights :: Matrix Double
  , _outputWeights :: Maybe (Matrix Double)
  -- ^ Trainable part of ESN
  , _par :: ESNParameters
  }

data ESNType = Plain
  | Leaky  -- ^ Leaky integrator ESN (not yet supported)

data ESNParameters = Par
  { _spectralRadius :: Double
  , _inputScaling :: V.Vector Double
  , _inputOffset :: V.Vector Double
  , _teacherScaling :: V.Vector Double
  , _teacherOffset :: V.Vector Double
  , _type_ :: ESNType
  }

-- | Default ESN parameters
par0 :: ESNParameters
par0 = Par { _spectralRadius = 0.5
           , _type_ = Plain
           -- NB: the four parameters below should be made compatible
           -- with _input-/_outputUnits (TODO).
           , _inputScaling = V.fromList [0.1, 0.1]
           , _inputOffset = V.fromList [0.0, 0.0]
           , _teacherScaling = V.singleton 0.3
           , _teacherOffset = V.singleton (-0.2)
           }

-- | Initialize a new echo state network of given dimensions
new
  :: StdGen
  -> ESNParameters
  -> (Int, Int, Int)
  -- ^ Input, internal, and output units number
  -> ESN
new g par (inu, iu, out) =
  let (g1, g1') = split g
      iw = H.randMatrix g1 (iu, inu) (-1, 1.0)
      (g2, g2') = split g1'
      fw = H.randMatrix g2 (iu, out) (-1, 1.0)
      (g3, _) = split g2'
      connectivity = min (10.0 / fromIntegral iu) 1
      rWeights = reservoirWeights g3 iu connectivity
      reservoir = Reservoir { _activation = tanh
                            , _weights = scalar (_spectralRadius par) * rWeights
                            }
      esn = ESN { _inputWeights = iw
                , _reservoir = reservoir
                , _feedbackWeights = fw
                , _outputWeights = Nothing
                , _par = par
                }
  in esn

reservoirWeights
  :: StdGen
  -> Int
  -> Double
  -> Matrix Double
reservoirWeights g iu connectivity = f $ split g
  where
    mg g' = let weights = H.randSparse g' (iu, iu) (-0.5, 0.5) connectivity
                eigenvalues = fst $ eig weights
                maxVal = V.maximum $ V.map magnitude eigenvalues
            -- Normalize using the maximal eigenvalues' magnitude
            in if maxVal < 1e-4
              then Nothing
              else Just $ cmap (/maxVal) weights

    -- Repeat until we obtain a valid matrix
    f (g1, g2) = fromMaybe (f $ split g2) (mg g1)

-- | Offline ESN training
learn
  :: ESN
  -> Int
  -- ^ Discard the first N points
  -> Matrix Double
  -- ^ Input matrix of features rows and observations columns
  -> Matrix Double
  -- ^ Desired output matrix of observations (data points) columns
  -> Either String ESN
learn esn forgetPts inp out = esn'
  where
    state' = scanNet esn inp (Just out) ?? (All, Drop forgetPts)
    teacher' = teacher (_par esn) out ?? (All, Drop forgetPts)
    esn' = case Learning.learn' state' teacher' of
      Nothing -> Left "Cannot create a readout matrix"
      w -> Right $ esn { _outputWeights = w }

-- | Either state matrix or signal y(t): depending if target sequence @out@
-- was provided or not.
-- If provided, the state matrix consists of two components:
-- 1. Input information
-- 2. Internal state, i.e. the _nonlinear transformation_
-- performed by a reservoir (dynamical system)
scanNet :: ESN
        -> Matrix Double
        -- ^ Input sequence
        -> Maybe (Matrix Double)
        -- ^ Target sequence (if training)
        -> Matrix Double
scanNet ESN { _par = Par { _inputScaling = is, _inputOffset = iof }
            , _inputWeights = w0
            , _reservoir = Reservoir { _activation = fnl, _weights = w1 }
            , _outputWeights = ow
            , _feedbackWeights = w2
            } inp out = st
  where
    inpScaled = inp * reshape 1 is
    inpWithOffset = inpScaled * reshape 1 iof

    internalUnits = rows w1

    -- Initial network state (zero)
    v0 = V.replicate internalUnits 0.0

    st = case out of
      -- Training
      Just out' ->
        let -- Empty state accumulator matrix
            m0 = tr $ reshape internalUnits $ vector []
            r = foldr (\(vIn, forcing) (m, v) ->
                          let v' = _netw v vIn forcing
                          in (m ||| reshape 1 v', v')
                      ) (m0, v0) $ zip (toColumns inpWithOffset) (toColumns out')
        in fst r
      -- Exploitation
      Nothing ->
        let readout = fromJust ow
            outputUnits = rows readout
            -- Empty y(t) signal accumulator
            m0 = tr $ reshape outputUnits $ vector []
            -- Initially, no feedback
            f0 = reshape 1 $ V.replicate outputUnits 0.0
            r = foldr (\vIn (m, (v, feedb)) ->
                         let v' = _netw v vIn feedb
                             y = readout #> v'
                         in (m ||| reshape 1 y, (v', y))
                      ) (m0, (v0, flatten f0)) $ toColumns inpWithOffset
        in fst r

    -- Network equation
    -- x(n + 1) = fnl[W * x(n) + Win * u(n + 1) + Wfb * y(n)]
    _netw statePrev iVec feedb = V.map fnl s
      where
        -- Propagate from the input layer
        s0 = w0 #> iVec

        -- Reservoir: transform the previous state
        s1 = w1 #> statePrev

        -- Teacher forcing or self-feedback
        s2 = w2 #> feedb

        s = V.zipWith (+) (V.zipWith (+) s0 s1) s2

-- | Drop @forgetPts@ and apply scaling and offset operations
-- to @ outp @ matrix of @ outDim x outPts @.
teacher
  :: ESNParameters
  -> Matrix Double
  -> Matrix Double
teacher Par { _teacherScaling = ts
            , _teacherOffset = to
            } outp = (d <> outp) + offset
  where
    d = diag ts
    offset = reshape 1 to

predict :: ESN
        -> Int
        -> Matrix Double
        -> Either String (Matrix Double)
predict esn@ESN { _outputWeights = ow
                , _par = Par { _teacherScaling = ts, _teacherOffset = to }
                } forgetPoints inpt =
  case ow of
    Nothing -> Left "Please train the ESN first"
    Just _ -> let y = scanNet esn inpt Nothing
                  y2 = y ?? (All, Drop forgetPoints)

              -- Scale back to original
              in Right $ (y2 - reshape 1 to) <> inv (diag ts)
