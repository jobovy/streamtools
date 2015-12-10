# The DF of a tidal stream peppered with impacts
import copy
import numpy
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
        self._nkicks= len(impactb)
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
