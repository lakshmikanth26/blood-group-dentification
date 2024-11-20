import pandas as pd
from tensorflow.keras.callbacks import History

# Assuming you have the history object from training
history = model.history  # This comes from the model.fit() function

# Save history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
