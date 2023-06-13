import numpy as np

from metadynminer.hills import Hills
from metadynminer.minima import Minima

class FEProfile:
    """
    Free energy profile is a visualization of differences between local 
    minima points during metadynamics simulation. If the values seem 
    to converge to a mean value of the difference, it suggests, 
    but not fully proof, that the FES did converge to the correct shape.
    
    Command:
    ```python
    fep = metadynminer.FEProfile(minima, hillsfile)
    ```
    
    Parameters:
    
    * minima = metadynminer.Minima object
    
    * hillsfile = metadynminer.Hills object
    
    """
    def __init__(self, minima: Minima, hills: Hills):
        self.cvs = minima.cvs
        self.res = minima.res
        self.minima = minima.minima
        self.periodic = minima.periodic
        self.heights = hills.get_heights()

        self.cv_name = minima.cv_name
        self.cv_min = minima.cv_min
        self.cv_max = minima.cv_max
        self.cv_per = minima.cv_per
        self.sigma = hills.sigma
        self.cv = hills.cv

        if len(minima.minima.shape) > 1:
            self.makefeprofile(hills)
        else: 
            raise ValueError(
                "There is only one local minimum on the free energy surface."
            )
        
        
    def makefeprofile(self, hills):
        """
        Internal method to calculate free energy profile.
        """
        hillslenght = len(hills.cv[:,0])
        
        if hillslenght < 256:
            profilelenght = hillslenght
            scantimes = np.array(range(hillslenght))
        else:
            profilelenght = 256
            scantimes = np.array(((hillslenght/(profilelenght))*np.array((range(1,profilelenght+1)))))
            scantimes -= 1
            scantimes = scantimes.astype(int)
        
        number_of_minima = self.minima.shape[0]
        
        self.feprofile = np.zeros((self.minima.Minimum.shape[0]+1))
        
        if self.cvs == 1:
            cv1min, cv1max = self.cv_min[0], self.cv_max[0]
            cv1_fes_range = self.cv_max[0] - self.cv_min[0]
            
            fes = np.zeros((self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,3]:
                    dist_cv1 = self.cv[:, 0][lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range

                    dp2 = dist_cv1**2/(2*self.sigma[:, 0][lasttime:time]**2)
                    tmp = np.zeros(self.cv[:, 0][lasttime:time].shape)
                    tmp[dp2<2.5] = self.heights[lasttime:time][dp2<2.5] * (np.exp(-dp2[dp2<2.5]) * 1.00193418799744762399 - 0.00193418799744762399)
                    fes[int((float(x)-cv1min)*self.res/cv1_fes_range)] -= tmp.sum()

                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2]))]-\
                                       fes[int(float(self.minima.iloc[0,2]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        elif self.cvs == 2:
            cv1min, cv1max = self.cv_min[0], self.cv_max[0]
            cv1_fes_range = self.cv_max[0] - self.cv_min[0]
            cv2min, cv2max = self.cv_min[1], self.cv_max[1]
            cv2_fes_range = self.cv_max[1] - self.cv_min[1]
            
            fes = np.zeros((self.res, self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,4]:
                    dist_cv1 = self.cv[:, 0][lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range
                    
                    for y in self.minima.iloc[:,5]:
                        dist_cv2 = self.cv[:, 1][lasttime:time]-float(y)
                        if self.periodic[1]:
                            dist_cv2[dist_cv2<-0.5*cv2_fes_range] += cv2_fes_range
                            dist_cv2[dist_cv2>+0.5*cv2_fes_range] -= cv2_fes_range
                    
                        dp2 = dist_cv1**2/(2*self.sigma[:, 0][lasttime:time]**2) + dist_cv2**2/(2*self.sigma[:, 1][lasttime:time]**2)
                        tmp = np.zeros(self.sigma[:, 0][lasttime:time].shape)
                        tmp[dp2<6.25] = self.heights[lasttime:time][dp2<6.25] * (np.exp(-dp2[dp2<6.25]) * 1.00193418799744762399 - 0.00193418799744762399)
                        fes[int((float(x)-cv1min)*self.res/cv1_fes_range),int((float(y)-cv2min)*self.res/cv2_fes_range)] -= tmp.sum()
                
                # save profile
                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2])),int(float(self.minima.iloc[m,3]))]-\
                                       fes[int(float(self.minima.iloc[0,2])),int(float(self.minima.iloc[0,3]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        elif self.cvs == 3:
            cv1min, cv1max = self.cv_min[0], self.cv_max[0]
            cv1_fes_range = self.cv_max[0] - self.cv_min[0]
            cv2min, cv2max = self.cv_min[1], self.cv_max[1]
            cv2_fes_range = self.cv_max[1] - self.cv_min[1]
            cv3min, cv3max = self.cv_min[2], self.cv_max[2]
            cv3_fes_range = self.cv_max[2] - self.cv_min[2]
            fes = np.zeros((self.res, self.res, self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,5]:
                    dist_cv1 = self.cv[:, 0][lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range
                    
                    for y in self.minima.iloc[:,6]:
                        dist_cv2 = self.cv[:, 1][lasttime:time]-float(y)
                        if self.periodic[1]:
                            dist_cv2[dist_cv2<-0.5*cv2_fes_range] += cv2_fes_range
                            dist_cv2[dist_cv2>+0.5*cv2_fes_range] -= cv2_fes_range
                        
                        for z in self.minima.iloc[:,7]:
                            dist_cv3 = self.cv[:, 2][lasttime:time]-float(z)
                            if self.periodic[2]:
                                dist_cv3[dist_cv3<-0.5*cv3_fes_range] += cv3_fes_range
                                dist_cv3[dist_cv3>+0.5*cv3_fes_range] -= cv3_fes_range
                    
                            dp2 = (dist_cv1**2/(2*self.sigma[:, 0][lasttime:time]**2) + 
                                   dist_cv2**2/(2*self.sigma[:, 1][lasttime:time]**2) + 
                                   dist_cv3**2/(2*self.sigma[:, 2][lasttime:time]**2))
                            tmp = np.zeros(self.cv[:, 0][lasttime:time].shape)
                            tmp[dp2<15.625] = (self.heights[lasttime:time][dp2<15.625] * 
                                               (np.exp(-dp2[dp2<15.625]) * 1.00193418799744762399 - 0.00193418799744762399))
                            fes[int((float(x)-cv1min)*self.res/cv1_fes_range),
                                int((float(y)-cv2min)*self.res/cv2_fes_range),
                                int((float(z)-cv3min)*self.res/cv3_fes_range)] -= tmp.sum()
                
                # save profile
                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2])),
                                           int(float(self.minima.iloc[m,3])),
                                           int(float(self.minima.iloc[m,4]))]-\
                                       fes[int(float(self.minima.iloc[0,2])),
                                           int(float(self.minima.iloc[0,3])),
                                           int(float(self.minima.iloc[0,4]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        else:
            print("Fes object doesn't have supported number of CVs.")
    
    def plot(self, name="FEprofile.png",image_size=[10,7], xlabel=None, ylabel=None, label_size=12, cmap="jet"):
        """
        Visualization function for FEP. 
        
        ```python
        fep.plot()
        ```
        
        
        Parameters:
        
        
        * name (default="FEProfile.png") = name for .png file to save the plot to
        
        * image_size (default=[10,7]) = list of two dimensions of the picture
        
        * xlabel (default="time (ps)")
        
        * ylabel (default="free energy difference (kJ/mol)") 
        
        * label_size (default=12) = size of labels
        
        * cmap (default="jet") = matplotlib colormap used for coloring the line of the minima
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        plt.figure(figsize=(image_size[0],image_size[1]))
        
        cmap=cm.get_cmap(cmap)
        
        #colors = cm.jet((self.minima.iloc[:,1].to_numpy()).astype(float)/\
        #                (np.max(self.minima.iloc[:,1].to_numpy().astype(float))))
        colors = cmap(np.linspace(0,1,self.minima.shape[0]))
        for m in range(self.minima.shape[0]):
            plt.plot(self.feprofile[:,0], self.feprofile[:,m+1], color=colors[m])

        if xlabel == None:
            plt.xlabel('time (ps)', size=label_size)
        else:
            plt.xlabel(xlabel, size=label_size)
        if ylabel == None:
            plt.ylabel('free energy difference (kJ/mol)', size=label_size)
        else:
            plt.ylabel(ylabel, size=label_size)
        plt.savefig(name)
