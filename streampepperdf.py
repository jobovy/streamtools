# The DF of a tidal stream peppered with impacts
import numpy
import galpy.df_src.streamdf
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

              impact_angle= angle offset from progenitor at which the impact occurred (rad) ([nimpact])

              timpact time since impact (scalar)

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed)

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
        timpact= kwargs.pop('timpact',1.)
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
        self._general_kick= GM is None or rs is None
        if self._general_kick and subhalopot is None:
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
        # Setup a streamgapdf object to setup the machinery to go between 
        # (x,v) and (Omega,theta) near the impact

        self._determine_nTrackIterations(kwargs.get('nTrackIterations',None))
        self._determine_deltaAngleTrackImpact(deltaAngleTrackImpact,timpact)
        self._determine_impact_coordtransform(self._deltaAngleTrackImpact,
                                              nTrackChunksImpact,
                                              timpact,impact_angle)
        # Compute \Delta Omega ( \Delta \theta_perp) and \Delta theta,
        # setup interpolating function
        self._determine_deltav_kick(impactb,subhalovel,
                                    GM,rs,subhalopot,
                                    nKickPoints)
        self._determine_deltaOmegaTheta_kick()
        # Then pass everything to the normal streamdf setup
        #self.nInterpolatedTrackChunks= 201 #more expensive now
        #super(streampepperdf,self)._determine_stream_track(nTrackChunks)
        #self._useInterp= useInterp
        #if interpTrack or self._useInterp:
        #    super(streampepperdf,self)._interpolate_stream_track()
        #    super(streampepperdf,self)._interpolate_stream_track_aA()
        #super(streampepperdf,self).calc_stream_lb()
        return None

