module RC.Helpers
  ( addBiases
  , randList
  , randMatrix
  , randSparse
  , hsigmoid
  ) where

import           System.Random
import           Data.List ( unfoldr )
import qualified Numeric.LinearAlgebra as LA

-- | Hard sigmoid
hsigmoid :: (Fractional a, Ord a)
         => (a, a, a)
         -- ^ Vertical scaling, width, offset
         -> a
         -> a
hsigmoid (β, width, offset) x = f
  where
    f | x < offset = 0.0
      | x < width = β * (x - offset)
      | otherwise = β * (width - offset)
{-# SPECIALISE hsigmoid :: (Double, Double, Double) -> Double -> Double #-}
{-# SPECIALISE hsigmoid :: (Float, Float, Float) -> Float -> Float #-}

-- | Prepend a row of ones
--
-- >>> addBiases $ (2><3) [20..26]
-- (3><3)
--  [  1.0,  1.0,  1.0
--  , 20.0, 21.0, 22.0
--  , 23.0, 24.0, 25.0 ]
addBiases :: LA.Matrix Double -> LA.Matrix Double
addBiases m = let no = LA.cols m
                  m' = LA.konst 1.0 (1, no)
              in m' LA.=== m

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
