Processed data to make it closer to stationary (may help model performance).
Daily timeframe used.
Prediction value is tommmorows close.
Long signal if prediction higher than todays close and short signal if lower than todays close.
Exit either on the close of the candle or if the stop loss is reached.
Parameters are various popular trading indicators.
Rolling train periods used so the model can adapt to recent market conditions.
Rolling distributions collected to decide on where to place stop loss (however a fixed stop of -3% was the final decision).
Risk free rate of 0% in Sharpe.
Am apprehensive about cumulative return, live conditions (slippage etc.) can diminish this.

Further work: extract prediction distribution, only take trade if a certain amount of the distribution is above\below todays close price. this may increase the accuracy.
