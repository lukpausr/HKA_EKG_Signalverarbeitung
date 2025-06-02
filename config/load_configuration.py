import os
import yaml

def load_configuration():

    """
    Load the configuration file based on the PC name.
    """
    # Determine the PC name based on the operating system
    # For Windows, use the 'COMPUTERNAME' environment variable
    # For other operating systems, use the 'nodename' from os.uname()
    pc_name = os.environ['COMPUTERNAME'] if os.name == 'nt' else os.uname().nodename

    # Print the detected PC name for verification
    print(f"PC Name: {pc_name}")

    # Check the PC name and set the path for the configuration file accordingly
    try:

        if pc_name == 'DESKTOP-LUKAS':
            config_path = "config/config_lukas.yaml"
        elif pc_name == 'Desktop-Mika':
            config_path = "config/config_mika.yaml"
        else:
            config_path = "config/config_lukas.yaml"

        # Load the configuration file
        with open(config_path) as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")

    except Exception as e:
    
        if pc_name == 'DESKTOP-LUKAS':
            config_path = "../config/config_lukas.yaml"
        elif pc_name == 'Desktop-Mika':
            config_path = "../config/config_mika.yaml"
        else:
            config_path = "../config/config_lukas.yaml"

        # Load the configuration file
        with open(config_path) as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")

    return config