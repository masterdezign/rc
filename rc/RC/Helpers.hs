module RC.Helpers where

import           System.Random
import           Data.List ( unfoldr )
import qualified Numeric.LinearAlgebra as LA

-- | Random matrix with elements in the range of [minVal; maxVal]
randMatrix
  :: StdGen
     -> (Int, Int)
     -- ^ Number of rows and columns
     -> (Double, Double)
     -- ^ Minimal and maximal values
     -> LA.Matrix Double
randMatrix seed (rows', cols') (minVal, maxVal) = (LA.reshape cols'. LA.vector) xs
  where
    xs = f <$> randList (rows' * cols') seed
    f x = (maxVal - minVal) * x + minVal

-- | Random sparse matrix
--
-- NB: at the moment, the matrix is stored in memory as
-- an ordinary (dense) matrix.
randSparse g (rows', cols') (minVal, maxVal) connectivity =
    LA.reshape cols' $ LA.vector xs
  where
    (g1, g2) = split g
    rlist = randList (rows' * cols')
    xs = zipWith f (rlist g1) (rlist g2)
    f lv rv | lv < connectivity = (maxVal - minVal) * rv + minVal
            | otherwise = 0.0

randList :: (Random a, Floating a) => Int -> StdGen -> [a]
randList n = take n. unfoldr (Just. random)
{-# SPECIALISE randList :: Int -> StdGen -> [Float] #-}
{-# SPECIALISE randList :: Int -> StdGen -> [Double] #-}
