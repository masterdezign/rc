-- |
-- = Single-node reservoir
--
-- In this project, we exploit an established analogy between
-- spatially extended and delay systems (refs. 1-3). That makes possible
-- to employ DDEs as a reservoir substrate.
--
-- 1. Arecchi, F. T., et al. “Two-Dimensional Representation of
--    a Delayed Dynamical System.” Physical Review A, vol. 45, no. 7,
--    Jan. 1992, doi:10.1103/physreva.45.r4225.
-- 2. Appeltant, L., et al. “Information Processing Using a Single
--    Dynamical Node as Complex System.” Nature Communications, vol. 2,
--    2011, p. 468., doi:10.1038/ncomms1476.
-- 3. Virtual Chimera States for Delayed-Feedback Systems - Laurent Larger,
--    Bogdan Penkovsky, Yuri Maistrenko. Physical Review Letters - 08 / 2013.

{-# LANGUAGE BangPatterns #-}
module RC.NTC.Reservoir
  ( Reservoir (..)
  , genReservoir
  ) where

import           Numeric.LinearAlgebra
import qualified Data.Vector.Storable as V
import qualified Numeric.DDE as DDE
import qualified Numeric.DDE.Model as DDEModel

import           RC.NTC.Types
import qualified RC.Helpers as H


-- | Substrate-specific low-level reservoir implementation
genReservoir :: DDEModel.Par -> Reservoir
genReservoir par@DDEModel.RC {
    DDEModel._filt = DDEModel.BandpassFiltering { DDEModel._tau = tau }
  } = Reservoir _r
  where
    _r sample = H.unflatten' nodes response
      where
        oversampling = 1 :: Int  -- No oversampling
        detuning = 1.0 :: Double  -- Delay detuning factor, 1 = no detuning
        nodes = rows sample
        delaySamples = round $ detuning * fromIntegral (oversampling * nodes)

        -- Matrix to timetrace
        trace1 = H.flatten' sample

        -- Duplicate the last element (DDE.integHeun2_2D consumes one extra input)
        trace = trace1 V.++ V.singleton (V.last trace1)

        -- Empirically chosen integration time step:
        -- twice faster than the system response time tau
        hStep = tau / 2

        !(_, !response) = DDE.integHeun2_2D delaySamples hStep (DDEModel.rhs par) (DDE.Input trace)

genReservoir _ = error "Unsupported DDE model"

