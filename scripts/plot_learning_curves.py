import re
import matplotlib.pyplot as plt

# Define the path to the log file
log_file_path = '../results/training.log'

# Read the log file content
log_content = []
with open(log_file_path, 'r') as file:
    log_content = file.readlines()

# Identify the starting point of the training log
start_line = 0
for i, line in enumerate(log_content):
    if "Epoch" in line:
        start_line = i
        break

# Define regex pattern to extract metrics from log lines
pattern = re.compile(
    r"loss: ([\d\.]+) - accuracy: ([\d\.]+) - val_loss: ([\d\.]+) - val_accuracy: ([\d\.]+)"
)

# Initialize lists to store the extracted data
epochs = []
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

# Extract data from the log file
for i, line in enumerate(log_content[start_line:], start=start_line):
    if "Epoch" in line and "/" in line:
        epoch_num = int(re.search(r"Epoch (\d+)", line).group(1))
        epochs.append(epoch_num)
    elif pattern.search(line):
        match = pattern.search(line)
        train_loss.append(float(match.group(1)))
        train_accuracy.append(float(match.group(2)))
        val_loss.append(float(match.group(3)))
        val_accuracy.append(float(match.group(4)))
        
        # Stop adding data if validation loss increases by more than 0.035
        if len(val_loss) > 1 and val_loss[-1] - val_loss[-2] > 0.035:
            break

# Align the lengths of the epochs and metrics arrays
min_length = min(len(epochs), len(train_loss), len(train_accuracy), len(val_loss), len(val_accuracy))

epochs = epochs[:min_length]
train_loss = train_loss[:min_length]
train_accuracy = train_accuracy[:min_length]
val_loss = val_loss[:min_length]
val_accuracy = val_accuracy[:min_length]

# Function to multiply epoch ticks by 10
def multiply_by_10(x, _):
    return int(x * 10)

# Plotting the learning curves
plt.figure(figsize=(12, 6))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(multiply_by_10))

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(multiply_by_10))

plt.tight_layout()
# save the plot to a file
plt.savefig('../results/learning_curves.png')
plt.show()

# Plotting joint plot for validation loss and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_loss, label='Validation Loss', color='blue', linestyle='--')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='green', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Validation Loss and Accuracy')
plt.legend()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(multiply_by_10))

plt.tight_layout()
plt.savefig('../results/joint_validation_plot.png')
plt.show()