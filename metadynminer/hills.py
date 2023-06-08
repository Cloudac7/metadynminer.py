import numpy as np

class Hills:
    """
    Object of Hills class are created for loading HILLS files, and obtaining the necessary information from them. 

    Hills files are loaded with command:
    ```python
    hillsfile = metadynminer.Hills()
    ```
    
    optional parameters:
    
    * name (default="HILLS") = string with name of HILLS file
    
    * ignoretime (default=True) = boolean, if set to False, it will save the time in the HILLS file;
                                if set to True, and timestep is not set, 
                                        each time value will be incremented by the same amount as the time of the first step.
                                        
    * timestep = numeric value of the time difference between hills, in picoseconds
    
    * periodic (default=[False, False]) = list of boolean values telling which CV is periodic.
    
    * cv_per (defaults = ([-numpy.pi, numpy.pi])) = \
        Tuple of lists containing two numeric values defining the periodicity of given CV. 
        Has to be provided for each periodic CV.
    """

    def __init__(
            self, 
            name="HILLS", 
            encoding="utf8", 
            ignoretime=True, 
            periodic=None, 
            cv_per=([-np.pi, np.pi]),
            timestep=None
    ):
        self.read(
            name, encoding, ignoretime, periodic, cv_per, timestep
        )
        self.hills_filename = name
    
    def read(
            self, 
            name="HILLS", 
            encoding="utf8", 
            ignoretime=True, 
            periodic=None, 
            cv_per=([-np.pi, np.pi]),
            timestep=None
    ):
        with open(name, 'r', encoding=encoding) as hills_file:
            first_line = hills_file.readline()
        columns = first_line.split() 
        number_of_columns_head = len(columns) - 2
        self.cvs = (number_of_columns_head - 3) // 2
        self.cv_name = columns[3:3+self.cvs]

        if periodic == None:
            periodic = [False for i in range(self.cvs)]
        self.periodic = periodic[:self.cvs]

        self.cv_per = cv_per
        
        t = 0
        self.hills = np.loadtxt(name, dtype=np.double)
        self.cv = self.hills[:, 1:1+self.cvs]
        self.cv_min = np.min(self.cv, axis=0) - 1e-8
        self.cv_max = np.max(self.cv, axis=0) + 1e-8
        for i in range(self.cvs):
            flag = 0
            if self.periodic[i]:
                try:
                    self.cv_min[i] = self.cv_per[flag][0]
                    self.cv_max[i] = self.cv_per[flag][1]
                    flag += 1
                except IndexError:
                    raise ValueError(
                        "cv_per has to be provided for each periodic CV"
                    )

        self.sigma = self.hills[:, 1+self.cvs:-2]
        self.heights = self.hills[:, -2]
        self.biasf = self.hills[:, -1]
        if ignoretime:
            if timestep == None:
                timestep = self.hills[0][0]
            self.hills[:, 0] = np.arange(
                timestep, timestep*(len(self.hills)+1), timestep
            )
    
    def get_cv(self, index=0):
        return self.hills[:, 1 + index]
    
    def get_cv_per(self, index=0):
        return self.cv_per[index]
    
    def get_periodic(self):
        return self.periodic
    
    def get_cv_name(self, index=0):
        return self.cv_name[index]
    
    def get_hills(self):
        return self.hills
    
    def get_number_of_cvs(self):
        return self.cvs
    
    def get_sigma(self, index=0):
        return self.sigma[:, index]
    
    def get_heights(self):
        return(self.heights)
