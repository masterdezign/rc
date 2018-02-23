import           Numeric.LinearAlgebra
import           System.Random ( mkStdGen )
import           Learning ( nrmse )

import           RC.NTC as RC

-- Get training data
takePast :: Int -> Matrix Double -> Matrix Double
takePast horizon xs = xs ?? (All, Take (len - horizon))
  where
    len = cols xs

-- Get teacher data
takeFuture :: Int -> Matrix Double -> Matrix Double
takeFuture horizon = (?? (All, Drop horizon))

splitAt' :: Int -> Matrix Double -> (Matrix Double, Matrix Double)
splitAt' i m = (m ?? (All, Take i), m ?? (All, Drop i))

main :: IO ()
main = do
  -- Load and transpose time series to predict
  dta <- tr <$> loadMatrix "examples/data/mg.txt"

  let splitRatio = 0.50  -- Train on 50% of data
      total = cols dta
      spl = round $ splitRatio * fromIntegral total
      -- Split the data into training and validation data sets
      (train, validate) = splitAt' spl dta

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
