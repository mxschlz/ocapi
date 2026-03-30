import os

def get_data_path():
  """Returns the correct file path to the EEG data based on the OS."""
  if os.name == 'nt':  # Windows
    data_path = 'G:\\Meine Ablage\\PhD\\data\\OCAPI\\'
  elif os.name == 'posix':  # Linux
    data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/'
  else:
    raise OSError("Unsupported operating system.")
  return data_path
