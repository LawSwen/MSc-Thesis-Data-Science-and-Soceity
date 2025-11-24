


''' This code plots the forecast of the Attention model.. It also plots the attention weights..
Not all the plots in this code are used in the thesis.. 
some were necessary for diagnostic and further studies to inform the studies outcomes '''

import all_imports as ai
ai.plt.rcParams.update({
    # Figure settings
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'figure.autolayout': True,
    
    # Font settings - INCREASED FOR PUBLICATION
    'font.size': 16,
    'font.family': 'DejaVu Sans',
    'font.weight': 'normal',
    
    # Axes settings - INCREASED
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.grid': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    
    # Grid settings
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    
    # Line settings - THICKER FOR VISIBILITY
    'lines.linewidth': 4,
    'lines.markersize': 10,
    
    # Legend settings - IMPROVED FOR PUBLICATION
    'legend.fontsize': 16,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.fancybox': True,
    'legend.shadow': True,
    
    # Tick settings - INCREASED
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    
    # Savefig settings - HIGH QUALITY
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'savefig.format': 'pdf'
})


class PredictionVisualizer:
    def __init__(self, y_true_original, y_pred_original, mask, forecast_horizon, target_col, scaler_y=None):
        self.y_true_original = y_true_original
        self.y_pred_original = y_pred_original
        self.mask = mask
        self.scaler_y = scaler_y
        self.forecast_horizon = forecast_horizon  # Fixed typo
        self.target_col = target_col

    def single_prediction_plot(self, sample_idx=0, figsize=(16, 8), save_path=None):
        """Plot single sample prediction with actual vs predicted"""
        actual = self.y_true_original[sample_idx].flatten()
        predicted = self.y_pred_original[sample_idx].flatten()
        sample_mask = self.mask[sample_idx].flatten()
        
        time_steps = ai.np.arange(len(actual))
        valid_mask = sample_mask > 0
        
        ai.plt.figure(figsize=figsize)
        ai.plt.plot(time_steps, actual, 'b-', label='Actual', alpha=0.8, linewidth=4)
        ai.plt.plot(time_steps[valid_mask], predicted[valid_mask], 'ro-', 
                label='Predicted', alpha=0.8, linewidth=3, markersize=10)
        
        invalid_mask = sample_mask == 0
        if ai.np.any(invalid_mask):
            ai.plt.fill_between(time_steps[invalid_mask], 
                           actual.min(), actual.max(), 
                           color='red', alpha=0.2, label='Masked')
        
    
        ai.plt.xlabel(f'Forecast_horizon:\n{self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'{self.target_col} Concentration', fontsize=30, fontweight='normal')
        ai.plt.legend(fontsize=16, loc='best', framealpha=0.9, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
    
    def scatter_plot(self, figsize=(12, 10), save_path=None):
        """Plot scatter plot of actual vs predicted values"""
        y_true_flat = self.y_true_original.reshape(-1)
        y_pred_flat = self.y_pred_original.reshape(-1)
        mask_flat = self.mask.reshape(-1)
        
        valid_indices = mask_flat > 0
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
        
        ai.plt.figure(figsize=figsize)
        ai.plt.scatter(y_true_valid, y_pred_valid, alpha=0.6, s=40)
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ai.plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)
        ai.plt.xlabel(f'Actual {self.target_col} Values', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'Predicted {self.target_col} Values\n {self.forecast_horizon}hours', fontsize=30, fontweight='normal')
        # ai.plt.title('Actual vs Predicted Scatter Plot', fontsize=22, fontweight='bold', pad=20)
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return y_true_valid, y_pred_valid
    
    def error_plot(self, figsize=(12, 8), save_path=None):
        """Plot distribution of prediction errors"""
        y_true_flat = self.y_true_original.reshape(-1)
        y_pred_flat = self.y_pred_original.reshape(-1)
        mask_flat = self.mask.reshape(-1)
        
        valid_indices = mask_flat > 0
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
        
        errors = y_true_valid - y_pred_valid
        
        ai.plt.figure(figsize=figsize)
        ai.plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ai.plt.xlabel('Prediction Error', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'{self.target_col}Concentration\n {self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return errors
    
    def absolute_error_plot(self, figsize=(12, 8), save_path=None):
        """Plot distribution of absolute errors"""
        y_true_flat = self.y_true_original.reshape(-1)
        y_pred_flat = self.y_pred_original.reshape(-1)
        mask_flat = self.mask.reshape(-1)
        
        valid_indices = mask_flat > 0
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
        
        errors = y_true_valid - y_pred_valid
        absolute_errors = ai.np.abs(errors)
        
        ai.plt.figure(figsize=figsize)
        ai.plt.hist(absolute_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ai.plt.xlabel('Absolute Error', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'Frequency \n{self.target_col} {self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
    
    def residuals_plot(self, figsize=(12, 8), save_path=None):
        """Plot residuals vs predicted values"""
        y_true_flat = self.y_true_original.reshape(-1)
        y_pred_flat = self.y_pred_original.reshape(-1)
        mask_flat = self.mask.reshape(-1)
        
        valid_indices = mask_flat > 0
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
        
        errors = y_true_valid - y_pred_valid
        
        ai.plt.figure(figsize=figsize)
        ai.plt.scatter(y_pred_valid, errors, alpha=0.6, s=40)
        ai.plt.axhline(y=0, color='r', linestyle='--', linewidth=3)
        ai.plt.xlabel(f'Predicted Values\n{self.target_col}: {self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.ylabel('Residuals', fontsize=30, fontweight='normal')
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return errors
    
    def error_percentage_plot(self, figsize=(12, 8), save_path=None):
        """Plot distribution of error percentages"""
        y_true_flat = self.y_true_original.reshape(-1)
        y_pred_flat = self.y_pred_original.reshape(-1)
        mask_flat = self.mask.reshape(-1)
        
        valid_indices = mask_flat > 0
        y_true_valid = y_true_flat[valid_indices]
        y_pred_valid = y_pred_flat[valid_indices]
        
        errors = y_true_valid - y_pred_valid
        epsilon = 1e-8
        error_percentage = (errors / (y_true_valid + epsilon)) * 100
        
        ai.plt.figure(figsize=figsize)
        ai.plt.hist(error_percentage, bins=50, alpha=0.7, edgecolor='black', color='green')
        ai.plt.axvline(x=0, color='r', linestyle='--', linewidth=3)
        ai.plt.xlabel(f'Error Percentage (%) \n{self.target_col} {self.forecast_horizon}\n', fontsize=30, fontweight='normal')
        ai.plt.ylabel('Frequency', fontsize=30, fontweight='normal')
     
        ai.plt.grid(True, alpha=0.3)
        
        # Fixed tick parameters
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(30)
            label.set_fontweight('normal')
            label.set_rotation(0)
            
        ai.plt.tight_layout()

        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return
    

class AttentionAnalyzer:
    def __init__(self, model, test_seq, target_col, forecast_horizon):
        self.model = model
        self.test_seq = test_seq
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        
    def get_attention_weights(self, sample_idx=0):
        """Extract attention weights for a specific sample"""
        test_inputs = {
            "X_enc": self.test_seq["X_enc"][sample_idx:sample_idx+1],
            "delta_enc": self.test_seq["delta_enc"][sample_idx:sample_idx+1],
            "decoder_input": self.test_seq["decoder_input"][sample_idx:sample_idx+1],
            "mask_enc": self.test_seq["mask_enc"][sample_idx:sample_idx+1]
        }
        
        predictions = self.model.predict(test_inputs, verbose=0)
        attention_weights = predictions["attention"][0]  # (T_dec, T_enc)
        forecast = predictions["forecast"][0]  # (T_dec, 1)
        
        return attention_weights, forecast
    
    def attention_heatmap_plot(self, sample_idx=0, figsize=(16, 12), save_path=None):
        """Plot attention weights as a heatmap"""
        attention_weights, _ = self.get_attention_weights(sample_idx)
        
        ai.plt.figure(figsize=figsize)
        im = ai.plt.imshow(attention_weights.T, aspect='auto', cmap='viridis', 
                          extent=[0, attention_weights.shape[0], 0, attention_weights.shape[1]])
        cbar = ai.plt.colorbar(im, label='Attention Weight')
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Attention Weight', fontsize=30, fontweight='normal')
        ai.plt.xlabel(f'Decoder Time Step (Future) - {self.forecast_horizon} hours', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'{self.target_col}: Encoder Time Step (Past)', fontsize=30, fontweight='normal')
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return attention_weights
    
    def attention_timeline_plot(self, sample_idx=0, figsize=(18, 8), save_path=None):
        """Plot attention evolution over decoder steps"""
        attention_weights, _ = self.get_attention_weights(sample_idx)
        
        ai.plt.figure(figsize=figsize)
        
        # Plot first 8 decoder steps for clarity
        for i in range(min(8, attention_weights.shape[0])):
            ai.plt.plot(attention_weights[i], 
                       label=f'Future t+{i}', alpha=0.8, linewidth=3)
        
        ai.plt.xlabel(f'Encoder Time Step (Past)\n{self.target_col}: {self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.ylabel(f'Attention Weight', fontsize=30, fontweight='normal')
        ai.plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, 
                     framealpha=0.9, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
    
    def attention_focus_plot(self, sample_idx=0, figsize=(14, 8), save_path=None):
        """Plot how focused the attention is over time"""
        attention_weights, _ = self.get_attention_weights(sample_idx)
        
        # Calculate attention focus (entropy-based)
        attention_entropy = -ai.np.sum(attention_weights * ai.np.log(attention_weights + 1e-8), axis=1)
        max_entropy = ai.np.log(attention_weights.shape[1])
        attention_focus = 1 - (attention_entropy / max_entropy)
        
        ai.plt.figure(figsize=figsize)
        ai.plt.plot(attention_focus, 'go-', linewidth=4, markersize=12)
        ai.plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, linewidth=3, label='Focus Threshold')
        ai.plt.xlabel(f'Decoder Time Step (Future) \n{self.target_col}:  {self.forecast_horizon} hours \n', fontsize=25, fontweight='normal')
        ai.plt.ylabel('Attention Focus\n (1 = focused, 0 = spread)\n', fontsize=25, fontweight='normal')
        ai.plt.legend(fontsize=16, framealpha=0.9, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return 
    
    def top_attended_steps_plot(self, sample_idx=0, top_k=8, figsize=(14, 8), save_path=None):
        """Plot the most attended encoder steps"""
        attention_weights, _ = self.get_attention_weights(sample_idx)
        
        total_attention = ai.np.sum(attention_weights, axis=0)
        top_indices = ai.np.argsort(total_attention)[-top_k:]
        top_values = total_attention[top_indices]
        
        ai.plt.figure(figsize=figsize)
        bars = ai.plt.bar(range(len(top_indices)), top_values, alpha=0.7, color='orange')
        ai.plt.xticks(range(len(top_indices)), 
                     [f't-{attention_weights.shape[1] - idx}' for idx in top_indices],
                     rotation=45, fontsize=20)
        ai.plt.xlabel(f'Encoder Time Step (Past) \n{self.target_col}: {self.forecast_horizon}hours\n', fontsize=25, fontweight='normal')
        ai.plt.ylabel('Total Attention Received', fontsize=30, fontweight='normal')
        # ai.plt.title(f'Top {top_k} Most Attended Past Steps - Sample {sample_idx}', 
        #             fontsize=22, fontweight='bold', pad=20)
        ai.plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, top_values):
            ai.plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{value:.2f}', ha='center', va='bottom', fontsize=16, fontweight='normal')
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
        
        return top_indices, top_values
    
    def attention_context_plot(self, sample_idx=0, figsize=(16, 10), save_path=None):
        """Plot attention in context with actual predictions"""
        attention_weights, forecast = self.get_attention_weights(sample_idx)
        
        # Get original scale data
        scaler_y = self.test_seq["scaler_y"]
        forecast_original = scaler_y.inverse_transform(forecast).flatten()
        target_original = scaler_y.inverse_transform(
            self.test_seq["decoder_target"][sample_idx]
        ).flatten()
        
        ai.plt.figure(figsize=figsize)
        
        # Plot predictions and highlight high-attention steps
        time_steps = ai.np.arange(len(target_original))
        ai.plt.plot(time_steps, target_original, 'b-', label='Actual', alpha=0.8, linewidth=4)
        ai.plt.plot(time_steps, forecast_original, 'ro-', label='Predicted', alpha=0.8, linewidth=3)
        
        # Highlight steps with strong attention to specific past steps
        max_attention_per_step = ai.np.max(attention_weights, axis=1)
        high_attention_threshold = 0.2
        
        for i in range(len(time_steps)):
            if max_attention_per_step[i] > high_attention_threshold:
                ai.plt.plot(time_steps[i], target_original[i], 'g*', 
                          markersize=20, alpha=0.8, label='High Attention' if i == 0 else "")
        
        ai.plt.xlabel(f'Future Time Step \n{self.target_col}:  {self.forecast_horizon} hours\n', fontsize=30, fontweight='normal')
        ai.plt.ylabel('Value (Original Scale)', fontsize=30, fontweight='normal')
        # ai.plt.title(f'Predictions with High-Attention Steps Highlighted - Sample {sample_idx}', 
        #             fontsize=30, fontweight='normal', pad=20)
        ai.plt.legend(fontsize=16, framealpha=0.9, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=6)
        
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
    
    def attention_statistics_plot(self, sample_idx=0, figsize=(18, 8), save_path=None):
        attention_weights, _ = self.get_attention_weights(sample_idx)
    
        # Separate numerical and text statistics
        numerical_stats = {
            'Avg Attention': ai.np.mean(attention_weights),
            'Max Attention': ai.np.max(attention_weights),
            'Attention Spread': ai.np.std(attention_weights),
            'Focus Steps': ai.np.sum(ai.np.max(attention_weights, axis=1) > 0.3)
        }
    
        text_stats = {
            'Most Attended': f't-{attention_weights.shape[1] - ai.np.argmax(ai.np.sum(attention_weights, axis=0))}'
        }
    
        ai.plt.figure(figsize=figsize)
    
        # Plot numerical statistics
        ai.plt.subplot(1, 2, 1)
        bars = ai.plt.bar(range(len(numerical_stats)), list(numerical_stats.values()), alpha=0.7)
        ai.plt.xticks(range(len(numerical_stats)), list(numerical_stats.keys()), rotation=0, fontsize=15)
        ai.plt.ylabel('Value', fontsize=30, fontweight='normal')
        ai.plt.xlabel(f"{self.target_col}:{self.forecast_horizon} hours", fontsize=30, fontweight="normal")
        # ai.plt.title('Numerical Attention Statistics', fontsize=30, fontweight='normal', pad=20)
        ai.plt.grid(True, alpha=0.3)
    
        # Add value annotations
        for i, (key, value) in enumerate(numerical_stats.items()):
            ai.plt.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=20, fontweight='normal')
    
        # Display text statistics
        ai.plt.subplot(1, 2, 2)
        ai.plt.axis('off')
        stats_text = "Additional Statistics:\n\n"
        for key, value in text_stats.items():
            stats_text += f"{key}: {value}\n"
    
        # Add some calculated insights
        total_attention = ai.np.sum(attention_weights, axis=0)
        most_attended_idx = ai.np.argmax(total_attention)
        stats_text += f"\nMost attended step: {most_attended_idx}\n"
        stats_text += f"Total decoder steps: {attention_weights.shape[0]}\n"
        stats_text += f"Total encoder steps: {attention_weights.shape[1]}"
    
        ai.plt.text(0.1, 0.5, stats_text, fontsize=30, transform=ai.plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
        # ai.plt.suptitle(f'Attention Statistics - Sample {sample_idx}', fontsize=22, fontweight='noraml')
        ai.plt.tight_layout()
        if save_path:
            ai.plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=600)
        ai.plt.show()
    
        # Combine all stats for return
        all_stats = {**numerical_stats, **text_stats}
        return 
    



class TrainingPlotter:
    """Publication-ready training history plotter"""
    
    def __init__(self, target_col, forecast_horizon, figsize=(12, 8)):
        self.figsize = figsize
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
    
    def plot_loss_history(self, history, save_path=None):
        """Plot only loss history"""
        history_dict = history.history if hasattr(history, 'history') else history
        
        ai.plt.figure(figsize=self.figsize)
        
        if 'loss' in history_dict:
            ai.plt.plot(history_dict['loss'], label='Training Loss', linewidth=4, color='#2E86AB')
        if 'val_loss' in history_dict:
            ai.plt.plot(history_dict['val_loss'], label='Validation Loss', linewidth=4, color='#A23B72')
        
        # ai.plt.title(f'Training and Validation Loss \t {self.target_col} forecast_horizon {self.forecast_horizon}', fontsize=24, fontweight='bold', pad=25)
        ai.plt.ylabel('Loss', fontsize=30, fontweight='normal')
        ai.plt.xlabel(f'Epoch \n {self.target_col} forecast_horizon {self.forecast_horizon}', fontsize=22, fontweight='normal')
        ai.plt.legend(fontsize=18, framealpha=0.95, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=20, width=3, length=8)
        
        ai.plt.tight_layout()
        
        if save_path:
            ai.plt.savefig(f"{save_path}_loss.pdf", format='pdf', bbox_inches='tight', dpi=600)
        
        ai.plt.show()
    
    def plot_mae_history(self, history, save_path=None):
        """Plot only MAE history"""
        history_dict = history.history if hasattr(history, 'history') else history
        
        mae_metrics = [key for key in history_dict.keys() if 'mae' in key.lower()]
        if not mae_metrics:
            print("No MAE metrics found")
            return
        
        ai.plt.figure(figsize=self.figsize)
        
        colors = ['#F18F01', '#C73E1D']
        for i, metric in enumerate(mae_metrics):
            color = colors[i % len(colors)]
            label_name = metric.replace('_', ' ').title().replace('Val ', 'Validation ')
            ai.plt.plot(history_dict[metric], label=label_name, linewidth=4, color=color)
        
        ai.plt.title(f'Masked MAE \t {self.target_col} forecast_horizon {self.forecast_horizon}', fontsize=30, fontweight='normal', pad=25)
        ai.plt.ylabel('MAE', fontsize=30, fontweight='normal')
        ai.plt.xlabel('Epoch', fontsize=30, fontweight='normal')
        ai.plt.legend(fontsize=18, framealpha=0.95, edgecolor='black')
        ai.plt.grid(True, alpha=0.3)
        
        ax = ai.plt.gca()
        ax.tick_params(axis='both', labelsize=30, width=3, length=8)
        
        ai.plt.tight_layout()
        
        if save_path:
            ai.plt.savefig(f"{save_path}_mae.pdf", format='pdf', bbox_inches='tight', dpi=600)
        
        ai.plt.show()