# The DF of a tidal stream peppered with impacts
import copy
import numpy
from scipy import integrate
import galpy.df_src.streamdf
import galpy.df_src.streamgapdf
class streampepperdf(galpy.df_src.streamdf.streamdf):
    """The DF of a tidal stream peppered with impacts"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the DF of a stellar stream peppered with impacts

        INPUT:

           streamdf args and kwargs

           Subhalo and impact parameters, for all impacts:

              impactb= impact parameter ([nimpact])

              subhalovel= velocity of the subhalo shape=(nimpact,3)

              impact_angle= angle offset from progenitor at which the impact occurred (at the impact time; in rad) ([nimpact])

              timpact time since impact ([nimpact])

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed); all kicks need to chose the same option

                 1( GM= mass of the subhalo ([nimpact])

                    rs= size parameter of the subhalo ([nimpact])

                 2( subhalopot= galpy potential object or list thereof (should be spherical); list of len nimpact (if the individual potentials are lists, need to give a list of lists)

           deltaAngleTrackImpact= (None) angle to estimate the stream track over to determine the effect of the impact [similar to deltaAngleTrack] (rad)

           nTrackChunksImpact= (floor(deltaAngleTrack/0.15)+1) number of chunks to divide the progenitor track in near the impact [similar to nTrackChunks]

           nKickPoints= (10xnTrackChunksImpact) number of points along the stream to compute the kicks at (kicks are then interpolated)

        OUTPUT:

           object

        HISTORY:

           2015-12-07 - Started based on streamgapdf - Bovy (UofT)

        """
        # Parse kwargs, everything except for timpact can be arrays
        impactb= kwargs.pop('impactb',[1.])
        subhalovel= kwargs.pop('subhalovel',numpy.array([[0.,1.,0.]]))
        impact_angle= kwargs.pop('impact_angle',[1.])
        GM= kwargs.pop('GM',None)
        rs= kwargs.pop('rs',None)
        subhalopot= kwargs.pop('subhalopot',None)
        if GM is None: GM= [None for b in impactb]
        if rs is None: rs= [None for b in impactb]
        if subhalopot is None: subhalopot= [None for b in impactb]
        timpact= kwargs.pop('timpact',[1.])
        deltaAngleTrackImpact= kwargs.pop('deltaAngleTrackImpact',None)
        nTrackChunksImpact= kwargs.pop('nTrackChunksImpact',None)
        nKickPoints= kwargs.pop('nKickPoints',None)
        # For setup later
        nTrackChunks= kwargs.pop('nTrackChunks',None)
        interpTrack= kwargs.pop('interpTrack',
                                galpy.df_src.streamdf._INTERPDURINGSETUP)
        useInterp= kwargs.pop('useInterp',
                              galpy.df_src.streamdf._USEINTERP)
        # Analytical Plummer or general potential?
        self._general_kick= GM[0] is None or rs[0] is None
        if self._general_kick and numpy.any([sp is None for sp in subhalopot]):
            raise IOError("One of (GM=, rs=) or subhalopot= needs to be set to specify the subhalo's structure")
        if self._general_kick:
            self._subhalopot= subhalopot
        else:
            self._GM= GM
            self._rs= rs
        # Now run the regular streamdf setup, but without calculating the
        # stream track (nosetup=True)
        kwargs['nosetup']= True
        super(streampepperdf,self).__init__(*args,**kwargs)
        # Setup streamgapdf objects to setup the machinery to go between 
        # (x,v) and (Omega,theta) near the impacts
        uniq_timpact= list(set(timpact))
        self._sgapdfs_coordtransform= {}
        for ti in uniq_timpact:
            sgapdf_kwargs= copy.deepcopy(kwargs)
            sgapdf_kwargs['timpact']= ti
            sgapdf_kwargs['impact_angle']= 1. # only affects a check
            if not self._leading: sgapdf_kwargs['impact_angle']= -1.
            sgapdf_kwargs['deltaAngleTrackImpact']= deltaAngleTrackImpact
            sgapdf_kwargs['nTrackChunksImpact']= nTrackChunksImpact
            sgapdf_kwargs['nKickPoints']= nKickPoints
            sgapdf_kwargs['GM']= GM[0] # Just to avoid error
            sgapdf_kwargs['rs']= rs[0] 
            sgapdf_kwargs['subhalopot']= subhalopot[0]
            sgapdf_kwargs['nokicksetup']= True
            self._sgapdfs_coordtransform[ti]=\
                galpy.df_src.streamgapdf.streamgapdf(*args,**sgapdf_kwargs)
        # Compute all kicks
        self._nKickPoints= nKickPoints
        self._determine_deltaOmegaTheta_kicks(impact_angle,impactb,subhalovel,
                                              timpact,GM,rs,subhalopot)
        return None

    def _determine_deltaOmegaTheta_kicks(self,impact_angle,impactb,subhalovel,
                                         timpact,GM,rs,subhalopot):
        """Compute the kicks in frequency-angle space for all impacts"""
        self._nKicks= len(impactb)
        self._sgapdfs= []
        # Go through in reverse impact order
        for kk in numpy.argsort(timpact):
            sgdf= copy.deepcopy(self._sgapdfs_coordtransform[timpact[kk]])
            # compute the kick using the pre-computed coordinate transformation
            sgdf._determine_deltav_kick(impact_angle[kk],impactb[kk],
                                        subhalovel[kk],
                                        GM[kk],rs[kk],subhalopot[kk],
                                        self._nKickPoints)
            sgdf._determine_deltaOmegaTheta_kick()
            self._sgapdfs.append(sgdf)
        # Store times
        self._timpact= numpy.sort(timpact)
        return None

    def pOparapar(self,Opar,apar):
        """
        NAME:

           pOparapar

        PURPOSE:

           return the probability of a given parallel (frequency,angle) offset pair

        INPUT:

           Opar - parallel frequency offset (array)

           apar - parallel angle offset along the stream (scalar)

        OUTPUT:

           p(Opar,apar)

        HISTORY:

           2015-12-09 - Written - Bovy (UofT)

        """
        if isinstance(Opar,(int,float,numpy.float32,numpy.float64)):
            Opar= numpy.array([Opar])
        apar= numpy.tile(apar,(len(Opar)))
        out= numpy.zeros(len(Opar))
        # Need to rewind each to each impact sequentially
        current_Opar= copy.copy(Opar)
        current_apar= copy.copy(apar)
        current_timpact= 0.
        remaining_indx= numpy.ones(len(Opar),dtype='bool')
        for kk,timpact in enumerate(self._timpact):
            # Compute ts and where they were at current impact for all
            ts= current_apar/current_Opar
            # Evaluate those that have ts < (timpact-current_timpact)
            afterIndx= remaining_indx*(ts < (timpact-current_timpact))
            out[afterIndx]=\
                super(streampepperdf,self).pOparapar(current_Opar[afterIndx],
                                                     current_apar[afterIndx],
                                                     tdisrupt=
                                                     self._tdisrupt\
                                                         -current_timpact)
            remaining_indx*= (True-afterIndx)
            if numpy.sum(remaining_indx) == 0: break
            # Compute Opar and apar just before this kick for next kick
            current_apar-= current_Opar*(timpact-current_timpact)
            dOpar_impact= self._sgapdfs[kk]._kick_interpdOpar(current_apar)
            current_Opar-= dOpar_impact
            current_timpact= timpact
        # Need one last evaluation for before the first kick
        if numpy.sum(remaining_indx) > 0:
            # Evaluate
            out[remaining_indx]=\
                super(streampepperdf,self).pOparapar(\
                current_Opar[remaining_indx],current_apar[remaining_indx],
                tdisrupt=self._tdisrupt-current_timpact)
        return out

    def density_par(self,dangle,tdisrupt=None):
        """
        NAME:

           density_par

        PURPOSE:

           calculate the density as a function of parallel angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

        OUTPUT:

           density(angle)

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if tdisrupt is None: tdisrupt= self._tdisrupt
        Tlow= 1./2./self._sigMeanOffset\
            -numpy.sqrt(1.-(1./2./self._sigMeanOffset)**2.)
        return integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                                  *(1+T*T)/(1-T*T)**2.\
                                  *self.pOparapar(T/(1-T*T)\
                                                      *numpy.sqrt(self._sortedSigOEig[2])\
                                                      +self._meandO,dangle),
                              Tlow,1.)[0]

    def meanOmega(self,dangle,oned=False,tdisrupt=None):
        """
        NAME:

           meanOmega

        PURPOSE:

           calculate the mean frequency as a function of angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

           oned= (False) if True, return the 1D offset from the progenitor (along the direction of disruption)

        OUTPUT:

           mean Omega

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if tdisrupt is None: tdisrupt= self._tdisrupt
        Tlow= 1./2./self._sigMeanOffset\
            -numpy.sqrt(1.-(1./2./self._sigMeanOffset)**2.)
        num=\
            integrate.quad(lambda T: (T/(1-T*T)\
                                          *numpy.sqrt(self._sortedSigOEig[2])\
                                          +self._meandO)\
                               *numpy.sqrt(self._sortedSigOEig[2])\
                               *(1+T*T)/(1-T*T)**2.\
                               *self.pOparapar(T/(1-T*T)\
                                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                                   +self._meandO,dangle),
                           Tlow,1.)[0]
        denom=\
            integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                               *(1+T*T)/(1-T*T)**2.\
                               *self.pOparapar(T/(1-T*T)\
                                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                                   +self._meandO,dangle),
                           Tlow,1.)[0]
        dO1D= num/denom
        if oned: return dO1D
        else:
            return self._progenitor_Omega+dO1D*self._dsigomeanProgDirection\
                *self._sigMeanSign

