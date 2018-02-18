{-# LANGUAGE TemplateHaskell #-}
module Main where

import           System.Random ( mkStdGen
                               , StdGen
                               , split
                               )
import           Control.Lens
import           Numeric.LinearAlgebra

import           RC.ESN as RC
import           RC.ESN ( ESN (..)
                        , Reservoir (..)
                        )
import           RC.Helpers as H

-- makeLenses ''ESN
-- makeLenses ''Reservoir

narma :: [Double] -> [Double]
narma s0 = s
  where
    s = 0.1: (zipWith (+) s2 s1)
    s1 = map ((+ 0.1). (0.7 *)) s0
    s2 = zipWith (*) s (map (1 -) s)

main :: IO ()
main = do
  let g = mkStdGen 1111
      (g1, g2) = split g
      esn = RC.new g1 RC.par0 (2, 30, 1)
      trainFraction = 0.5
      forgetPoints = 100
      nPoints = 1000

      -- Generate the input signal
      inpt1 = replicate nPoints 1
      inpt2 = randList nPoints g2
      fromList' = reshape nPoints. vector
      inpt = (fromList' inpt1) ===
               (fromList' inpt2) :: Matrix Double

      -- Generate the output signal
      outp = fromList' $ take nPoints $ narma inpt2

  -- Train the network
  case RC.learn esn forgetPoints inpt outp of
    Left s -> error s
    Right esn' -> do
        -- print $ _outputWeights esn'
        case RC.predict esn' forgetPoints inpt of
          Left s -> error s
          Right outp' -> mapM_ print $ toList (flatten outp')
