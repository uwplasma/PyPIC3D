import toml

# Christopher Woolford
# Dec 30, 2024
# This script contains my implementation of perfectly electrical conducting (PEC) boundary conditions for the 3D PIC code.

def grab_pec_keys(config):
    """
    Extracts keys from a configuration dictionary that start with 'pec'.
    Args:
        config (dict): A dictionary containing configuration keys and values.
    Returns:
        list: A list of keys from the configuration dictionary that start with 'pec'.
    """
    pec_keys = []
    for key in config.keys():
        if key[:2] == 'pec':
            pec_keys.append(key)
    return pec_keys

def read_pec_boundaries_from_toml(toml_file):
    """
    Reads PEC boundaries from a TOML file and returns a list of PEC objects.

    Args:
        toml_file (str): Path to the TOML file containing PEC boundary definitions.

    Returns:
        list: A list of PEC objects created from the TOML file.
    """

    config = toml.load(toml_file)
    pec_keys = grab_pec_keys(config)

    pecs = []

    for toml_key in pec_keys:
        name  = config[toml_key]['name']
        shape = config[toml_key]['shape']
        xmin  = config[toml_key]['xmin']
        xmax  = config[toml_key]['xmax']
        ymin  = config[toml_key]['ymin']
        ymax  = config[toml_key]['ymax']
        zmin  = config[toml_key]['zmin']
        zmax  = config[toml_key]['zmax']

        pecs.append( PEC(name, xmin, xmax, ymin, ymax, zmin, zmax) )

    return pecs


class PEC:
    def __init__(self, name, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize a new PEC (Perfect Electric Conductor) object.

        Parameters:
        name (str): The name of the PEC object.
        xmin (float): The minimum x-coordinate of the PEC boundary.
        xmax (float): The maximum x-coordinate of the PEC boundary.
        ymin (float): The minimum y-coordinate of the PEC boundary.
        ymax (float): The maximum y-coordinate of the PEC boundary.
        zmin (float): The minimum z-coordinate of the PEC boundary.
        zmax (float): The maximum z-coordinate of the PEC boundary.
        """
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def apply_pec(self, Ex, Ey, Ez):
        """
        Apply Perfect Electric Conductor (PEC) boundary conditions to the electric field components.

        This method sets the electric field components (Ex, Ey, Ez) to zero at the boundaries
        defined by xmin, xmax, ymin, ymax, zmin, and zmax.

        Parameters:
        Ex (ndarray): The x-component of the electric field.
        Ey (ndarray): The y-component of the electric field.
        Ez (ndarray): The z-component of the electric field.

        Returns:
        tuple: A tuple containing the modified electric field components (Ex, Ey, Ez) with PEC boundary conditions applied.
        """
        Ey = Ey.at[ [self.xmin, self.xmax], :, : ].set(0)
        Ez = Ez.at[ [self.xmin, self.xmax], :, : ].set(0)

        Ex = Ex.at[ [self.ymin, self.ymax], :, : ].set(0)
        Ez = Ez.at[ [self.ymin, self.ymax], :, : ].set(0)

        Ex = Ex.at[ [self.zmin, self.zmax], :, : ].set(0)
        Ey = Ey.at[ [self.zmin, self.zmax], :, : ].set(0)

        return Ex, Ey, Ez