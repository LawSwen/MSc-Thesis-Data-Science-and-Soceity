import all_imports as ai

def prepare_data(data, target_col):
   ''' prepare the data for modeling'''
   data = data[[target_col, 'hour', 
       'month', 'Weekend', 'is_holiday', 'season']].copy()
   
   ''' take the sine and cosine of hour and 
   month to capture the cyclical nature of time    '''
   
   data["day_of_week"] = ai.pd.to_datetime(data.index).dayofweek
   data['hour_sin'] = ai.np.sin(2 * ai.np.pi * data['hour'] / 24)
   data['hour_cos'] = ai.np.cos(2 * ai.np.pi * data['hour'] / 24)
   data['month_sin'] = ai.np.sin(2 * ai.np.pi * data['month'] / 12)
   data['month_cos'] = ai.np.cos(2 * ai.np.pi * data['month'] / 12)

   ''' turn categorical variables into dummies '''

   season_dummies = ai.pd.get_dummies(data['season'], prefix='season').astype(int)
   dow_dummies = ai.pd.get_dummies(data['day_of_week'], prefix='dow').astype(int)

   '''' concatenate all the data together '''

   df = ai.pd.concat([data, season_dummies, dow_dummies], axis=1)

   '''drop the original categorical columns '''
   df = df.drop(columns=['hour', 'month', 'season', 'day_of_week']).copy()  

   return df


def create_sequences_keras(data, input_seq_len, output_seq_len, target_indices):
    enc_inputs, dec_inputs, dec_targets, scalers = [], [], [], []
    num_series, total_timesteps, _ = data.shape

    ''' This loop is design to be flexible for multivariate series, however, 
    in our case, we are working with univariate series'''

    for t in range(total_timesteps - input_seq_len - output_seq_len + 1):
        enc_input = data[:, t:t+input_seq_len, :]

        ''' this sesssion of the code implement teacher forcing mechanism where the decoder input
        is the last true value....'''
        dec_input = data[:, t+input_seq_len-1:t+input_seq_len+output_seq_len-1, target_indices]
        dec_target = data[:, t+input_seq_len:t+input_seq_len+output_seq_len, target_indices]

        enc_scaled, dec_in_scaled, dec_tgt_scaled, scaler_list = [], [], [], []
        for i in range(num_series):
            scaler_input = ai.StandardScaler()
            scaler_target = ai.StandardScaler()

            '''Fit_transform on the encoder_transform on each series...'''

            enc_scaled.append(scaler_input.fit_transform(enc_input[i]))
            dec_in_scaled.append(scaler_target.fit_transform(dec_input[i].reshape(-1, 1)))
            dec_tgt_scaled.append(scaler_target.transform(dec_target[i].reshape(-1, 1)))
            scaler_list.append(scaler_target)

        enc_inputs.append(ai.np.stack(enc_scaled))
        dec_inputs.append(ai.np.stack(dec_in_scaled))
        dec_targets.append(ai.np.stack(dec_tgt_scaled))
        scalers.append(scaler_list)

    return {
        'enc_inputs': ai.np.stack(enc_inputs),
        'dec_inputs': ai.np.stack(dec_inputs),
        'dec_targets': ai.np.stack(dec_targets),
        'scalers': scalers
    }

def prepare_sequence_data(df, input_seq_len=24, output_seq_len=24, target_indices=[0], split_ratio=(0.7, 0.85)):
    """
    Prepares encoder/decoder inputs and targets for training, validation, and testing.

    Parameters:
    - df: pandas DataFrame of time series data
    - input_seq_len: length of encoder input sequence
    - output_seq_len: length of decoder output sequence
    - target_indices: list of column indices to forecast
    - split_ratio: tuple of (train_end, val_end) as fractions of total length

    Returns:
    - X_enc_train, X_dec_train, Y_train
    - X_enc_val, X_dec_val, Y_val
    - X_enc_test, X_dec_test, Y_test
    - scalers_test
    """

    # Drop timestamp if present and convert to numpy
    data_np = df.to_numpy()
    total_len = len(data_np)
    train_end = int(total_len * split_ratio[0])
    val_end = int(total_len * split_ratio[1])
    target_column_name = df.columns[target_indices[0]]


    # Chronological split
    train_data = data_np[:train_end][ai.np.newaxis, :, :]
    val_data = data_np[train_end:val_end][ai.np.newaxis, :, :]
    test_data = data_np[val_end:][ai.np.newaxis, :, :]

    # Sequence generation
    train_seq = create_sequences_keras(train_data, input_seq_len, output_seq_len, target_indices)
    val_seq = create_sequences_keras(val_data, input_seq_len, output_seq_len, target_indices)
    test_seq = create_sequences_keras(test_data, input_seq_len, output_seq_len, target_indices)

    # Extract and squeeze
    X_enc_train = train_seq['enc_inputs'].squeeze(axis=1)
    X_dec_train = train_seq['dec_inputs'].squeeze(axis=1)
    Y_train = train_seq['dec_targets'].squeeze(axis=1)

    X_enc_val = val_seq['enc_inputs'].squeeze(axis=1)
    X_dec_val = val_seq['dec_inputs'].squeeze(axis=1)
    Y_val = val_seq['dec_targets'].squeeze(axis=1)

    X_enc_test = test_seq['enc_inputs'].squeeze(axis=1)
    X_dec_test = test_seq['dec_inputs'].squeeze(axis=1)
    Y_test = test_seq['dec_targets'].squeeze(axis=1)
    scalers_test = test_seq['scalers']

    return (X_enc_train, X_dec_train, Y_train,
            X_enc_val, X_dec_val, Y_val,
            X_enc_test, X_dec_test, Y_test,
            scalers_test, target_column_name, output_seq_len)


class KalmanImpute:
    def __init__(self, data, mask, target_column="CO", noise_level=0.1):
        """
        Parameters:
        - data: pd.DataFrame with all features
        - mask: pd.DataFrame or pd.Series with binary mask (1 = observed, 0 = missing)
        - target_column: column to impute
        - noise_level: Kalman process noise for 'level' component
        """
        self.data = data.copy()
        self.mask = mask.copy()
        self.target_column = target_column
        self.noise_level = noise_level

    def convert_zeros_to_nan(self):
        series = self.data[self.target_column].copy()
        mask_series = self.mask[self.target_column] if isinstance(self.mask, ai.pd.DataFrame) else self.mask
        series[mask_series == 0] = ai.np.nan
        return series

    def kalman_impute_series(self, series):
        data = series.values.reshape(1, -1)
        smoother = ai.KalmanSmoother(
            component='level',
            component_noise={'level': self.noise_level}
        )
        smoother.smooth(data)
        return ai.pd.Series(smoother.smooth_data[0], index=series.index, name=series.name)

    def run(self):
        # Create series with NaN only at masked (missing) positions
        sparse_series = self.convert_zeros_to_nan()
        
        # Get Kalman smoothed series (this imputes all points)
        kalman_series = self.kalman_impute_series(sparse_series)
        
        # Create the final result: keep original values where observed, use Kalman only for missing
        df_imputed = self.data.copy()
        mask_series = self.mask[self.target_column] if isinstance(self.mask, ai.pd.DataFrame) else self.mask
        
        # Only replace the masked (missing) values with Kalman imputations
        missing_indices = mask_series == 0
        df_imputed.loc[missing_indices, self.target_column] = kalman_series[missing_indices]
        
        return df_imputed