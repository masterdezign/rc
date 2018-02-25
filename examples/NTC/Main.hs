import           Numeric.LinearAlgebra
import           System.Random ( mkStdGen )
import           Learning ( nrmse )

import           RC.NTC as RC

-- Get training data
takePast :: Element a => Int -> Matrix a -> Matrix a
takePast horizon xs = xs ?? (All, Take (len - horizon))
  where
    len = cols xs

-- Get target data
takeFuture :: Element a => Int -> Matrix a -> Matrix a
takeFuture horizon = (?? (All, Drop horizon))

-- Manipulate the data matrix: horizontal split
splitAtRatio :: Element a => Double -> Matrix a -> (Matrix a, Matrix a)
splitAtRatio ratio m = splitAt' spl m
  where
    total = cols m
    spl = round $ ratio * fromIntegral total

splitAt' :: Element a => Int -> Matrix a -> (Matrix a, Matrix a)
splitAt' i m = (m ?? (All, Take i), m ?? (All, Drop i))

main :: IO ()
main = do
  -- Load and transpose time series to predict
  dta <- tr <$> loadMatrix "examples/data/mg.txt"

  -- Train on 50% of data
  let (train, validate) = splitAtRatio 0.50 dta

  -- Configure a new NTC network
  let p = RC.par0 { RC._inputWeightsRange = (0.1, 0.3) }
      g = mkStdGen 1111
      ntc = RC.new g p (1, 1000, 1)

  -- 20 steps ahead prediction horizon
  let horizon = 20

  let train' = takePast horizon train  -- Past series
      trainTeacher = takeFuture horizon train  -- Predicted series

  let forgetPts = 300  -- Washout

  -- Train
  case RC.learn ntc forgetPts train' trainTeacher of
    Left s -> error s
    Right ntc' -> do
      let target = (takeFuture horizon validate) ?? (All, Drop forgetPts)

      -- Predict
      case RC.predict ntc' forgetPts (takePast horizon validate) of
        Left s -> error s
        Right prediction -> do
          let tgt' = flatten target
              predic' = flatten prediction
              err = nrmse tgt' predic'

          putStrLn $ "Error: " ++ show err

          let result = (tr target) ||| (tr prediction)

          saveMatrix "examples/NTC/result.txt" "%g" result
