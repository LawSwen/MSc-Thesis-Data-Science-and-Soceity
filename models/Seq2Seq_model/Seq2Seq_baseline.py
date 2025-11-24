import all_imports as ai

class Encoder(ai.Model):
    def __init__(self, units, num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.gru_layers = [ai.GRU(units, return_sequences=(i < num_layers - 1)) for i in range(num_layers)]
        self.bn_layers = [ai.BatchNormalization() for _ in range(num_layers)]
        self.dropout_layers = [ai.Dropout(dropout_rate) for _ in range(num_layers)]

    def call(self, x, training=False):
        for gru, bn, drop in zip(self.gru_layers, self.bn_layers, self.dropout_layers):
            x = gru(x)
            x = bn(x, training=training)
            x = drop(x, training=training)
        return x
'''This is the decoder session: params: units, output_dim(one), defualt_dropoutrate:
However, these can be optimize..'''

class Decoder(ai.Model):
    def __init__(self, units, output_dim, dropout_rate=0.2):
        super().__init__()
        self.gru = ai.GRU(units, return_sequences=True)
        self.bn = ai.BatchNormalization()
        self.dropout = ai.Dropout(dropout_rate)
        self.dense = ai.TimeDistributed(ai.Dense(output_dim))

    def call(self, context, dec_inputs, training=False):
        context_repeated = ai.RepeatVector(dec_inputs.shape[1])(context)
        x = ai.Concatenate(axis=-1)([dec_inputs, context_repeated])
        x = self.gru(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense(x)


''' This is the model class that compile both the encoder and decoder sessions
params: enc_units, dec_units, output_dim(one), defualt_dropout_rate(0.2)'''
class SequenceToSequenceModel(ai.Model):
    def __init__(self, enc_units, dec_units, output_dim, dropout_rate=0.2):
        super().__init__()
        self.encoder = Encoder(enc_units, num_layers=2, dropout_rate=dropout_rate)
        self.decoder = Decoder(dec_units, output_dim, dropout_rate)

    def call(self, inputs, training=False):
        x, dec_inputs = inputs
        context = self.encoder(x, training=training)
        return self.decoder(context, dec_inputs, training=training)
    


''' This class computes error matrices and plot forecast.. 
params: model, sparsity_level:{Defualt:baseline"}, scaler_list:{takes the sclar for inverse conversion}'''
class ForecastPipeline:
    def __init__(self, model, sparsity_level="Baseline", scaler_list=None):
        self.model = model
        self.scalers_test = scaler_list
        self.sparsity_level = sparsity_level

    def train_model(self, X_enc_train, X_dec_train, Y_train,
                    X_enc_val, X_dec_val, Y_val,
                    batch_size=64, epochs=100, patience=10, checkpoint_path=None):
        callbacks = [ai.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        if checkpoint_path:
            callbacks.append(ai.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True))

        history = self.model.fit(
            x=[X_enc_train, X_dec_train],
            y=Y_train,
            validation_data=([X_enc_val, X_dec_val], Y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate_model(self, X_enc_test, X_dec_test, Y_test):
        loss, mae = self.model.evaluate([X_enc_test, X_dec_test], Y_test, verbose=0)
        Y_pred_scaled = self.model.predict([X_enc_test, X_dec_test], verbose=0)

        Y_pred = ai.np.array([
            self.scalers_test[i][0].inverse_transform(Y_pred_scaled[i])
            for i in range(len(Y_pred_scaled))
        ])
        Y_true = ai.np.array([
            self.scalers_test[i][0].inverse_transform(Y_test[i])
            for i in range(len(Y_test))
        ])
        return Y_true, Y_pred

    def compute_metrics(self, Y_true, Y_pred, target_column_name):
        epsilon = 1e-8
        mask = ai.np.abs(Y_true.flatten()) > epsilon
        mae_score = ai.mean_absolute_error(Y_true.flatten(), Y_pred.flatten())
        mse_score = ai.mean_squared_error(Y_true.flatten(), Y_pred.flatten())
        rmse_score = ai.np.sqrt(mse_score)

        '''an episilon is use to avoid division by zero whhich can lead to very inflated MAPE sscore.'''
        mape_score = ai.np.mean(ai.np.abs((Y_true.flatten()[mask] - Y_pred.flatten()[mask]) /
           (Y_true.flatten()[mask] + epsilon))) * 100


        smape_score = ai.np.mean(
            2 * ai.np.abs(Y_pred.flatten() - Y_true.flatten()) /
            (ai.np.abs(Y_pred.flatten()) + ai.np.abs(Y_true.flatten()) + epsilon)
        ) * 100
        print(f"\n📊 Evaluation on Inverse-Transformed Test Set:")
        print(f"MAE   = {mae_score:.4f}")
        print(f"MAPE  = {mape_score:.4f}%")
        print(f"SMAPE = {smape_score:.4f}%")
        print(f"MSE   = {mse_score:.4f}")
        print(f"RMSE  = {rmse_score:.4f}")
        print(f"Sparsity_level: {self.sparsity_level}")

        metrices = {"MAE": round((mae_score), 4), "MAPE %": round(float(mape_score), 4), "SMAPE %": round(float(smape_score), 4), "RMSE": round(float(rmse_score), 4),
                    "Sparsity_Type": self.sparsity_level,
                    "Polluntant_name":target_column_name}

        return metrices

      

    def plot_forecast(self, Y_true, Y_pred, target_column_name, output_seq_len, sample_index=0,
                  tick_fontsize=40, tick_width=1, tick_length=5, save_path=None):
    
        fig, ax = ai.plt.subplots(figsize=(20, 8))
        ax.plot(Y_true[sample_index], label=f'Actual {target_column_name}', marker='o')
        ax.plot(Y_pred[sample_index], label=f'Predicted {target_column_name}', marker='x', linestyle='-')

        # ax.set_title(f"Forecast vs Actual on Test Sample ({target_column_name})")
        ax.set_xlabel(f"Forecast_horizon({output_seq_len})hours", fontsize=40)
        ax.set_ylabel(f"{target_column_name} Concentration", fontsize=40)

        # ✅ Tick customization
        ax.tick_params(axis='both', labelsize=tick_fontsize, width=tick_width, length=tick_length)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_fontsize)
            label.set_fontweight('normal')
            label.set_rotation(0)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
        
        ai.plt.show()

        ''' plot_error_distribution: plots both a scatter plot and a histrogram if show_histogram=True,
        else, only plot a scatter plot, saves plots it save_path=True'''
    def plot_error_distribution(self, Y_true, Y_pred, target_column_name, show_histogram=True,
                            tick_fontsize=40, tick_width=1, tick_length=5, save_path=None):
   
        errors = (Y_pred - Y_true).flatten()

        if show_histogram:
            fig, axes = ai.plt.subplots(1, 2, figsize=(20, 10))
            '''histogram plot'''

        # 📉 Histogram
            axes[0].hist(errors, bins=30, color='skyblue', edgecolor='black')
            axes[0].set_title('Error Distribution')
            axes[0].set_xlabel('Prediction Error', fontsize=40)
            axes[0].set_ylabel('Frequency', fontsize=40)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.8)

        # 📊 Scatter plot
            axes[1].scatter(Y_true.flatten(), Y_pred.flatten(), alpha=0.5)
            min_val = min(Y_true.min(), Y_pred.min())
            max_val = max(Y_true.max(), Y_pred.max())

            axes[1].plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=8, label='Align Prediction', zorder=10)
            # axes[1].plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()],
            #          'r--', linewidth=5, label='Align Prediction')
            axes[1].set_title('Predicted vs Actual')
            axes[1].set_xlabel(f'Actual Values ({target_column_name})', fontsize=40)
            axes[1].set_ylabel(f'Predicted Values ({target_column_name})', fontsize=40)
            axes[1].legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.8)
            axes[1].grid(True, alpha=0.3)

        # ✅ Tick customization
            for ax in axes:
                ax.tick_params(axis='both', labelsize=tick_fontsize, width=tick_width, length=tick_length)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(tick_fontsize)
                    label.set_fontweight('normal')
                    label.set_rotation(0)

        else:
            fig, ax = ai.plt.subplots(figsize=(20, 8))
            ax.scatter(Y_true.flatten(), Y_pred.flatten(), alpha=0.5)
            ax.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
            # ax.set_title('Predicted vs Actual')
            ax.set_xlabel(f'Actual Values ({target_column_name})', fontsize=40)
            ax.set_ylabel(f'Predicted Values ({target_column_name})', fontsize=40)
            ax.legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.8)
            ax.tick_params(axis='x', labelsize=20, labelrotation=0)
            ax.tick_params(axis='y', labelsize=20, labelrotation=0)
            ax.grid(True, alpha=0.3)

        # ✅ Tick customization
            ax.tick_params(axis='both', labelsize=tick_fontsize, width=tick_width, length=tick_length)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(tick_fontsize)
                label.set_fontweight('normal')
                label.set_rotation(0)
        ai.plt.tight_layout()        
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
        ai.plt.show()
  