import all_imports as ai

''' This code prepares the Sequence for the attention model'''

class DataPreparationAttention:

    '''params-> dataframe, target_colom, (start_time, end_time -> both detetermines the time
    range for sparsity in the series)'''
    def __init__(self, df, target_col, start_hour=1, end_hour=7):
        self.df = df.copy()
        self.target_col = target_col
        self.start_hour = start_hour
        self.end_hour = end_hour

    def prepare_features(self):
        df = self.df[[self.target_col, 'hour', 'month', 'Weekend', 'is_holiday', 'season']].copy()
        df["day_of_week"] = ai.pd.to_datetime(df.index).dayofweek

        df['hour_sin'] = ai.np.sin(2 * ai.np.pi * df['hour'] / 24)
        df['hour_cos'] = ai.np.cos(2 * ai.np.pi * df['hour'] / 24)
        df['month_sin'] = ai.np.sin(2 * ai.np.pi * df['month'] / 12)
        df['month_cos'] = ai.np.cos(2 * ai.np.pi * df['month'] / 12)

        season_dummies = ai.pd.get_dummies(df['season'], prefix='season').astype(int)
        dow_dummies = ai.pd.get_dummies(df['day_of_week'], prefix='dow').astype(int)

        df_final = ai.pd.concat([df, season_dummies, dow_dummies], axis=1)
        df_final.drop(columns=['hour', 'month', 'season', 'day_of_week'], inplace=True)

        assert not df_final.isnull().any().any(), "❌ Missing values in prepared features"
        return df_final

    def apply_block_sparsity(self, year):
        df_prepared = self.prepare_features()

        # Create mask for entire timesteps (1D array)
        mask_series = ai.pd.Series(1.0, index=df_prepared.index, name='mask')

        # Identify rows in the target year and target hours
        is_target_year = df_prepared.index.year == year
        is_target_hour = (df_prepared.index.hour >= self.start_hour) & (df_prepared.index.hour <= self.end_hour)
        mask_condition = is_target_year & is_target_hour

        # Apply sparsity: zero out target values and update mask
        df_sparse = df_prepared.copy()
        df_sparse.loc[mask_condition, self.target_col] = 0.0
        mask_series.loc[mask_condition] = 0.0

        # Final check
        assert len(df_sparse) == len(mask_series), "❌ Row mismatch between data and mask"
        ai.logging.info("Data Preparation Done!")

        return df_sparse, mask_series
    

def extract_X_y_M_delta(df_raw, target_col, sparse_year=2020, start_hour=1, end_hour=7):
    prep = DataPreparationAttention(df_raw, target_col, start_hour, end_hour)
    df_sparse, mask_df = prep.apply_block_sparsity(year=sparse_year)

    X_seq = df_sparse.drop(columns=[target_col]).values
    y_seq = df_sparse[[target_col]].values
    
    M_seq = mask_df.values.astype(ai.np.float32)

    delta_seq = ai.np.zeros_like(M_seq, dtype=ai.np.float32)
    last_obs = 0

    '''Delta is computed to quantify the temporal decay from the current time step to the last valid (unmasked)
      observation, enabling the model to account for the  decay in information over mask time'''

    for t in range(len(M_seq)):
        if M_seq[t]:
            delta_seq[t] = t - last_obs
            last_obs = t
        else:
            delta_seq[t] = t - last_obs

    assert X_seq.shape[0] == y_seq.shape[0] == M_seq.shape[0] == delta_seq.shape[0], "Time dimension mismatch"

    return X_seq, y_seq, M_seq, delta_seq, target_col



class SequencePreprocessor:
    def __init__(self, input_seq_len, output_seq_len, target_col, train_ratio=0.7, val_ratio=0.15):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_col = target_col
        self.scaler_x = ai.StandardScaler()
        self.scaler_y = ai.StandardScaler()

    def _assert_shapes(self, X, Y, M, delta_t):
        assert len(X) == len(Y), f"X and Y length mismatch: {len(X)} vs {len(Y)}"
        assert len(X) == len(M), f"X and M length mismatch: {len(X)} vs {len(M)}"
        assert len(X) == len(delta_t), f"X and Δt length mismatch: {len(X)} vs {len(delta_t)}"

    def split_data(self, X, Y, M, delta_t):
        self._assert_shapes(X, Y, M, delta_t)
        total_len = len(X)
        train_end = int(total_len * self.train_ratio)
        val_end = train_end + int(total_len * self.val_ratio)

        return {
            "train": {
                "X": X[:train_end],
                "Y": Y[:train_end],
                "M": M[:train_end],
                "delta_t": delta_t[:train_end]
            },
            "val": {
                "X": X[train_end:val_end],
                "Y": Y[train_end:val_end],
                "M": M[train_end:val_end],
                "delta_t": delta_t[train_end:val_end]
            },
            "test": {
                "X": X[val_end:],
                "Y": Y[val_end:],
                "M": M[val_end:],
                "delta_t": delta_t[val_end:]
            }
        }

    def scale_data(self, train_data, val_data, test_data):
        Y_train = train_data["Y"].reshape(-1, 1) if train_data["Y"].ndim == 1 else train_data["Y"]
        Y_val = val_data["Y"].reshape(-1, 1) if val_data["Y"].ndim == 1 else val_data["Y"]
        Y_test = test_data["Y"].reshape(-1, 1) if test_data["Y"].ndim == 1 else test_data["Y"]

        X_train_scaled = self.scaler_x.fit_transform(train_data["X"])
        Y_train_scaled = self.scaler_y.fit_transform(Y_train)
        X_val_scaled = self.scaler_x.transform(val_data["X"])
        Y_val_scaled = self.scaler_y.transform(Y_val)
        X_test_scaled = self.scaler_x.transform(test_data["X"])
        Y_test_scaled = self.scaler_y.transform(Y_test)

        assert X_train_scaled.shape == train_data["X"].shape, "X_train scaling shape mismatch"
        assert Y_train_scaled.shape == Y_train.shape, "Y_train scaling shape mismatch"

        return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test_scaled, Y_test_scaled

    def create_sequences(self, X_scaled, Y_scaled, M, delta_t):
        enc_inputs, delta_inputs, dec_inputs, dec_targets = [], [], [], []
        mask_enc, mask_dec = [], []
        total_timesteps = X_scaled.shape[0]

        if Y_scaled.ndim == 1:
            Y_scaled = Y_scaled.reshape(-1, 1)

        for t in range(total_timesteps - self.input_seq_len - self.output_seq_len + 1):
            x_enc = X_scaled[t:t+self.input_seq_len]
            delta_enc = delta_t[t:t+self.input_seq_len].reshape(-1, 1)

            ''' Here we apply teacher forcing logic'''

            y_dec_in = Y_scaled[t+self.input_seq_len-1:t+self.input_seq_len+self.output_seq_len-1]
            y_dec_out = Y_scaled[t+self.input_seq_len:t+self.input_seq_len+self.output_seq_len]

            if y_dec_in.ndim == 1:
                y_dec_in = y_dec_in.reshape(-1, 1)
            if y_dec_out.ndim == 1:
                y_dec_out = y_dec_out.reshape(-1, 1)

            assert x_enc.shape == (self.input_seq_len, X_scaled.shape[1]), "x_enc shape mismatch"
            assert y_dec_in.shape == (self.output_seq_len, Y_scaled.shape[1]), "y_dec_in shape mismatch"
            assert y_dec_out.shape == (self.output_seq_len, Y_scaled.shape[1]), "y_dec_out shape mismatch"

            mask_enc.append(M[t:t+self.input_seq_len].astype(ai.np.float32))
            mask_dec.append(M[t+self.input_seq_len:t+self.input_seq_len+self.output_seq_len].astype(ai.np.float32))

            enc_inputs.append(x_enc)
            delta_inputs.append(delta_enc)
            dec_inputs.append(y_dec_in)
            dec_targets.append(y_dec_out)

        return {
            "X_enc": ai.np.array(enc_inputs),
            "delta_enc": ai.np.array(delta_inputs),
            "decoder_input": ai.np.array(dec_inputs),
            "decoder_target": ai.np.array(dec_targets),
            "mask_enc": ai.np.array(mask_enc),
            "mask_dec": ai.np.array(mask_dec),
        }
    ''' Here we process our sequence and store all our
      values for tranining, validation and testing..'''

    def process(self, X, Y, M, delta_t):
        split = self.split_data(X, Y, M, delta_t)
        X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test_scaled, Y_test_scaled = self.scale_data(
            split["train"], split["val"], split["test"]
        )

        train_seq = self.create_sequences(X_train_scaled, Y_train_scaled, split["train"]["M"], split["train"]["delta_t"])
        val_seq = self.create_sequences(X_val_scaled, Y_val_scaled, split["val"]["M"], split["val"]["delta_t"])
        test_seq = self.create_sequences(X_test_scaled, Y_test_scaled, split["test"]["M"], split["test"]["delta_t"])

        for seq in [train_seq, val_seq, test_seq]:
            seq["scaler_x"] = self.scaler_x
            seq["scaler_y"] = self.scaler_y

        return train_seq, val_seq, test_seq, self.output_seq_len, self.target_col