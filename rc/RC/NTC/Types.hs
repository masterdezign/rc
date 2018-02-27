module RC.NTC.Types
  ( Reservoir (..)
  , NTCParameters (..)
  , NTC (..)
  ) where

import           System.Random ( StdGen )
import           Numeric.LinearAlgebra ( Matrix )

-- | Reservoir abstraction.
--
-- Reservoir (a recurrent neural network) is an essential
-- component in the NTC framework.
newtype Reservoir = Reservoir { _transform :: Matrix Double -> Matrix Double }

-- | Customizable NTC parameters
data NTCParameters = Par
  { _preprocess :: Matrix Double -> Matrix Double
    -- ^ Modify data before masking (e.g. compression)
  , _inputWeightsRange :: (Double, Double)  -- ^ Input weights (mask) range
  , _inputWeightsGenerator :: StdGen -> (Int, Int) -> (Double, Double) -> Matrix Double
  , _postprocess :: Matrix Double -> Matrix Double
  -- ^ Modify data before training or prediction (e.g. add biases)
  , _reservoirModel :: Reservoir
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
