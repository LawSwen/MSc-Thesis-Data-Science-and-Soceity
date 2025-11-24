
import all_imports as ai

class TimeAwareGRUCell(ai.tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        '''params-> units, **kwargs'''
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units

        self.gru_cell = ai.tf.keras.layers.GRUCell(units)
        self.decay = ai.tf.keras.layers.Dense(units, activation='sigmoid')

    def build(self, input_shape):
        assert input_shape[-1] >= 2, "Input must have at least 2 dims (features + delta_t)"
        feature_dim = input_shape[-1] - 1
        
        self.gru_cell.build((None, feature_dim))
        self.decay.build((None, 1))
        
        self.built = True

    def call(self, inputs, states):
        x_t = inputs[:, :-1]
        delta_t = inputs[:, -1:]
        h_prev = states[0]

        gamma = self.decay(delta_t)
        h_tilde = gamma * h_prev

        output, new_state = self.gru_cell(x_t, [h_tilde])
        
        return output, new_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config
    

class MaskedAttention(ai.tf.keras.layers.Layer):
    '''Implement mask attentionnon'''
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = ai.tf.keras.layers.Dense(units)
        self.W2 = ai.tf.keras.layers.Dense(units)
        self.V = ai.tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.W1.build((None, self.units))
        self.W2.build((None, self.units))
        self.V.build((None, self.units))
        
        self.built = True

    def call(self, h_t, memory, mask):
        h_t_expanded = ai.tf.expand_dims(h_t, axis=1)

        score = self.V(ai.tf.nn.tanh(self.W1(memory) + self.W2(h_t_expanded)))
        score = ai.tf.squeeze(score, axis=-1)

        score = ai.tf.where(mask > 0, score, ai.tf.fill(ai.tf.shape(score), -1e9))

        attention_weights = ai.tf.nn.softmax(score, axis=1)

        context = ai.tf.reduce_sum(ai.tf.expand_dims(attention_weights, -1) * memory, axis=1)

        '''returns attention weights and context vector'''
        
        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

class AttentionSequence(ai.tf.keras.layers.Layer):

    '''apply attention over the entire sequece through a loop'''
    def __init__(self, attention_layer, **kwargs):
        super().__init__(**kwargs)
        self.attention = attention_layer

    def build(self, input_shape):
        self.built = True

    def call(self, decoder_hidden, encoder_output, mask):
        batch_size = ai.tf.shape(decoder_hidden)[0]
        T_dec = ai.tf.shape(decoder_hidden)[1]

        context_seq = ai.tf.TensorArray(dtype=ai.tf.float32, size=T_dec, dynamic_size=False)
        weight_seq = ai.tf.TensorArray(dtype=ai.tf.float32, size=T_dec, dynamic_size=False)

        def loop_body(t, context_seq, weight_seq):
            h_t = decoder_hidden[:, t, :]
            context_t, weights_t = self.attention(h_t, encoder_output, mask)
            context_seq = context_seq.write(t, context_t)
            weight_seq = weight_seq.write(t, weights_t)
            return t + 1, context_seq, weight_seq

        _, context_seq, weight_seq = ai.tf.while_loop(
            cond=lambda t, *_: t < T_dec,
            body=loop_body,
            loop_vars=(0, context_seq, weight_seq),
            parallel_iterations=1
        )

        '''stack the context vector and the wieghts'''

        context_stack = ai.tf.transpose(context_seq.stack(), [1, 0, 2])
        weight_stack = ai.tf.transpose(weight_seq.stack(), [1, 0, 2])
        
        return context_stack, weight_stack
    



class ModelBuilder:
    def __init__(self, train_seq, units=64, config=None):
        self.train_seq = train_seq
        self.units = units
        self.config = config if config else {}

    def build_model(self):
        X_enc = self.train_seq["X_enc"]
        delta_enc = self.train_seq["delta_enc"]
        decoder_input = self.train_seq["decoder_input"]

        batch_size, T_enc, input_dim = X_enc.shape
        T_dec = decoder_input.shape[1]
        output_dim = decoder_input.shape[2]

        enc_input = ai.tf.keras.Input(shape=(T_enc, input_dim), name="X_enc")
        delta_input = ai.tf.keras.Input(shape=(T_enc, 1), name="delta_enc")
        dec_input = ai.tf.keras.Input(shape=(T_dec, output_dim), name="decoder_input")
        mask_enc_input = ai.tf.keras.Input(shape=(T_enc,), name="mask_enc")

        encoder_combined = ai.tf.keras.layers.Concatenate(axis=-1)([enc_input, delta_input])
        
        # Add dropout layers if specified in config
        input_dropout = self.config.get('input_dropout', 0.0)
        if input_dropout > 0:
            encoder_combined = ai.tf.keras.layers.Dropout(input_dropout)(encoder_combined)
        
        encoder_rnn = ai.tf.keras.layers.RNN(
            TimeAwareGRUCell(self.units), 
            return_sequences=True,
            return_state=False
        )
        encoder_output = encoder_rnn(encoder_combined, mask=mask_enc_input)

        # Add dropout after encoder if specified
        recurrent_dropout = self.config.get('recurrent_dropout', 0.0)
        if recurrent_dropout > 0:
            encoder_output = ai.tf.keras.layers.Dropout(recurrent_dropout)(encoder_output)

        decoder_hidden = ai.tf.keras.layers.GRU(
            self.units, 
            return_sequences=True,
            return_state=False
        )(dec_input)

        # Add dropout after decoder if specified
        if recurrent_dropout > 0:
            decoder_hidden = ai.tf.keras.layers.Dropout(recurrent_dropout)(decoder_hidden)

        attention_layer = MaskedAttention(self.units)
        attention_seq = AttentionSequence(attention_layer)
        
        context_vectors, attention_weights = attention_seq(
            decoder_hidden, encoder_output, mask_enc_input
        )

        combined = ai.tf.keras.layers.Concatenate()([decoder_hidden, context_vectors])
        
        # Add dropout before output if specified
        output_dropout = self.config.get('output_dropout', 0.0)
        if output_dropout > 0:
            combined = ai.tf.keras.layers.Dropout(output_dropout)(combined)
        
        forecast_output = ai.tf.keras.layers.TimeDistributed(
            ai.tf.keras.layers.Dense(1), name="forecast"
        )(combined)

        model = ai.tf.keras.Model(
            inputs=[enc_input, delta_input, dec_input, mask_enc_input],
            outputs={"forecast": forecast_output, "attention": attention_weights}
        )
        
        return model
    

class MSEPerTimestep(ai.tf.keras.losses.Loss):
    def __init__(self, name="mse_per_timestep"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mse_per_timestep = ai.tf.square(y_true - y_pred)
        return ai.tf.reduce_mean(mse_per_timestep, axis=-1)
    

class ModelEvaluator:
    def __init__(self, model, test_seq, scaler_y, forecast_horizon, sparsity_type, target_col):
        self.model = model
        self.test_seq = test_seq
        self.scaler_y = scaler_y
        self.forecast_horizon = forecast_horizon
        self.sparsity_type = sparsity_type
        self.target_col = target_col
        
    def get_predictions(self):
        """Get model predictions on test data"""
        test_inputs = {
            "X_enc": self.test_seq["X_enc"],
            "delta_enc": self.test_seq["delta_enc"],
            "decoder_input": self.test_seq["decoder_input"],
            "mask_enc": self.test_seq["mask_enc"]
        }
        
        predictions = self.model.predict(test_inputs, verbose=1)
        return predictions["forecast"]
    
    def inverse_transform(self, scaled_data):
        """Convert scaled data back to original scale using only the target scaler"""
        original_shape = scaled_data.shape
        flattened = scaled_data.reshape(-1, 1)
        inverted = self.scaler_y.inverse_transform(flattened)
        return inverted.reshape(original_shape)
    
    
    def compute_metrics(self, y_true_original, y_pred_original, mask):
        y_true_flat = y_true_original.reshape(-1)
        y_pred_flat = y_pred_original.reshape(-1)
        mask_flat = mask.reshape(-1)
    
    # Apply mask to get valid values only
        valid_indices = mask_flat > 0.5  # Use threshold for binary mask
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
    
    # Calculate MAE and RMSE
        mae = ai.mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = ai.np.sqrt(ai.mean_squared_error(y_true_valid, y_pred_valid))
    
    # Safe MAPE calculation - avoid division by zero
        non_zero_mask = ai.np.abs(y_true_valid) > 1e-8  # Only use non-zero true values
        if ai.np.sum(non_zero_mask) > 0:
            y_true_nonzero = y_true_valid[non_zero_mask]
            y_pred_nonzero = y_pred_valid[non_zero_mask]
            mape = ai.np.mean(ai.np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        else:
            mape = 0.0  # or float('nan') if you prefer
    
        metrics = {
            'mae': float(round(mae, 4)),
            'rmse': float(round(rmse, 4)),
            'mape': float(round(mape, 4)),
            "forecast_horizon": self.forecast_horizon,
            "sparsity_type": self.sparsity_type,
            "target_col": self.target_col
            }
    
        return metrics, y_true_valid, y_pred_valid
    
    def evaluate(self):
        """Complete evaluation pipeline: predictions + metrics"""
        predictions_scaled = self.get_predictions()
        
        predictions_original = self.inverse_transform(predictions_scaled)
        targets_original = self.inverse_transform(self.test_seq["decoder_target"])
        mask = self.test_seq["mask_dec"]
        
        metrics, y_true_valid, y_pred_valid = self.compute_metrics(
            targets_original, predictions_original, mask
        )
        
        print("📊 MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print("=" * 50)

       
        outcome =  { 'predictions_original': predictions_original,
            'targets_original': targets_original,
            'mask': mask,
            'y_true_valid': y_true_valid,
            'y_pred_valid': y_pred_valid}
           
        
        return metrics, outcome
    
 