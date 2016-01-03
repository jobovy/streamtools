# The DF of a tidal stream peppered with impacts
import copy
import numpy
from scipy import integrate, special, stats, optimize
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

           spline_order= (3) order of the spline to interpolate the kicks with

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
        spline_order= kwargs.pop('spline_order',3)
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
        self._uniq_timpact= list(set(timpact))
        self._sgapdfs_coordtransform= {}
        for ti in self._uniq_timpact:
            sgapdf_kwargs= copy.deepcopy(kwargs)
            sgapdf_kwargs['timpact']= ti
            sgapdf_kwargs['impact_angle']= impact_angle[0]#only affects a check
            if not self._leading: sgapdf_kwargs['impact_angle']= -1.
            sgapdf_kwargs['deltaAngleTrackImpact']= deltaAngleTrackImpact
            sgapdf_kwargs['nTrackChunksImpact']= nTrackChunksImpact
            sgapdf_kwargs['nKickPoints']= nKickPoints
            sgapdf_kwargs['spline_order']= spline_order
            sgapdf_kwargs['GM']= GM[0] # Just to avoid error
            sgapdf_kwargs['rs']= rs[0] 
            sgapdf_kwargs['subhalopot']= subhalopot[0]
            sgapdf_kwargs['nokicksetup']= True
            self._sgapdfs_coordtransform[ti]=\
                galpy.df_src.streamgapdf.streamgapdf(*args,**sgapdf_kwargs)
        self._gap_leading=\
            self._sgapdfs_coordtransform[timpact[0]]._gap_leading
        # Compute all kicks
        self._nKickPoints= nKickPoints
        self._spline_order= spline_order
        self._determine_deltaOmegaTheta_kicks(impact_angle,impactb,subhalovel,
                                              timpact,GM,rs,subhalopot)
        return None

    def simulate(self,rate=1.):
        """
        NAME:
        
           simulate

        PURPOSE:

           simulate a set of impacts

        INPUT:

           rate= (1.) 

        OUTPUT:

           (none; just sets up the instance for the new set of impacts)
        
        HISTORY

           2015-12-14 - Written - Bovy (UofT)

        """
        # Sample impact parameters
        # angle, just use Weibull for now
        angles=\
            numpy.cumsum(numpy.random.weibull(2.,
                                              size=int(numpy.ceil(rate))))\
                                              /rate
        angles= angles[angles < 1.]*self._deltaAngleTrack
        # Times and rewind impact angles
        timpacts= [self._uniq_timpact[0] for a in angles]
        impact_angles= numpy.array(\
            [a-t*super(streampepperdf,self).meanOmega(a,oned=True)
             for a,t in zip(angles,timpacts)])
        # BOVY: FOR NOW=>
        impact_angles[impact_angles <= 0.]= 0.1
        if not self._gap_leading: impact_angles*= -1.
        # Keep GM and rs the same for now
        GMs= numpy.array([self._GM[0] for a in impact_angles])
        rss= numpy.array([self._rs[0] for a in impact_angles])
        # impact b
        impactbs= numpy.random.uniform(size=len(impact_angles))**0.5\
            *2.*rss
        # velocity
        subhalovels= numpy.random.normal(scale=150./self._Vnorm,
                                         size=(len(impact_angles),3))
        # Setup
        self.set_impacts(impact_angle=impact_angles,
                         impactb=impactbs,
                         subhalovel= subhalovels,
                         timpact=timpacts,
                         GM=GMs,rs=rss)
        return None

    def set_impacts(self,**kwargs):
        """
        NAME:

           set_impacts

        PURPOSE:

           Setup a new set of impacts

        INPUT:

           Subhalo and impact parameters, for all impacts:

              impactb= impact parameter ([nimpact])

              subhalovel= velocity of the subhalo shape=(nimpact,3)

              impact_angle= angle offset from progenitor at which the impact occurred (at the impact time; in rad) ([nimpact])

              timpact time since impact ([nimpact])

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed); all kicks need to chose the same option

                 1( GM= mass of the subhalo ([nimpact])

                    rs= size parameter of the subhalo ([nimpact])

                 2( subhalopot= galpy potential object or list thereof (should be spherical); list of len nimpact (if the individual potentials are lists, need to give a list of lists)


        OUTPUT:
        
           (none; just sets up new set of impacts)

        HISTORY:

           2015-12-14 - Written - Bovy (UofT)

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
        # Run through previous setup to determine delta Omegas
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
                                        self._nKickPoints,self._spline_order)
            sgdf._determine_deltaOmegaTheta_kick(self._spline_order)
            self._sgapdfs.append(sgdf)
        # Store impact parameters
        sortIndx= numpy.argsort(numpy.array(timpact))
        self._timpact= numpy.array(timpact)[sortIndx]
        self._impact_angle= numpy.array(impact_angle)[sortIndx]
        self._impactb= numpy.array(impactb)[sortIndx]
        self._subhalovel= numpy.array(subhalovel)[sortIndx]
        self._GM= numpy.array(GM)[sortIndx]
        self._rs= numpy.array(rs)[sortIndx]
        self._subhalopot= [subhalopot[ii] for ii in sortIndx]
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
            afterIndx= remaining_indx*(ts < (timpact-current_timpact))\
                *(ts >= 0.)
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

    def _density_par(self,dangle,tdisrupt=None,approx=True):
        """The raw density as a function of parallel angle
        approx= use faster method that directly integrates the spline
        representations"""
        if tdisrupt is None: tdisrupt= self._tdisrupt
        if approx:
            return self._density_par_approx(dangle,tdisrupt)
        else:
            smooth_dens= super(streampepperdf,self)._density_par(dangle)
            return integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                                      *(1+T*T)/(1-T*T)**2.\
                                      *self.pOparapar(T/(1-T*T)\
                                                          *numpy.sqrt(self._sortedSigOEig[2])\
                                                          +self._meandO,dangle)/\
                                      smooth_dens,
                                  -1.,1.,
                                  limit=100,epsabs=1.e-06,epsrel=1.e-06)[0]\
                                  *smooth_dens

    def _density_par_approx(self,dangle,tdisrupt,_return_array=False,
                            *args):
        """Compute the density as a function of parallel angle using the 
        spline representations"""
        if len(args) == 0:
            ul,da,ti,c0,c1= self._approx_pdf(dangle)
        else:
            ul,da,ti,c0,c1= args
            ul= copy.copy(ul)
        # Find the lower limit of the integration interval
        lowbindx,lowx,edge= self.minOpar(dangle,False,True,ul,da,ti,c0,c1)
        ul[lowbindx-1]= ul[lowbindx]-lowx
        # Integrate each interval
        out= (0.5/c1*(special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                                      *(ul-c0-self._meandO))\
                          -special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])
                                       *(ul-c0-self._meandO
                                         -c1*(ul-numpy.roll(ul,1))))))
        if _return_array:
            return out
        out= numpy.sum(out[lowbindx:])
        # Add integration to infinity
        out+= 0.5*(1.+special.erf((self._meandO-ul[-1])\
                                      /numpy.sqrt(2.*self._sortedSigOEig[2])))
        # Add integration to edge if edge
        if edge:
            out+= 0.5*(special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                                       *(ul[0]-self._meandO))\
                           -special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])
                                        *(dangle/tdisrupt-self._meandO)))
        return out

    def _approx_pdf(self,dangle):
        """Internal function to return all of the parameters of the (approximat) p(Omega,angle)"""
        # First construct the breakpoints for the last impact for this dangle,
        # and start on the upper limits
        Oparb= (dangle-self._sgapdfs[0]._kick_interpdOpar_poly.x)\
            /self._timpact[0]
        ul= Oparb[::-1]
        # Array of previous pw-poly coeffs and breaks
        pwpolyBreak= self._sgapdfs[0]._kick_interpdOpar_poly.x[::-1]
        pwpolyCoeff0= numpy.append(\
            self._sgapdfs[0]._kick_interpdOpar_poly.c[-1],0.)[::-1]
        pwpolyCoeff1= numpy.append(\
            self._sgapdfs[0]._kick_interpdOpar_poly.c[-2],0.)[::-1]
        # Arrays for propagating the lower and upper limits through the impacts
        da= numpy.ones_like(ul)*dangle
        ti= numpy.ones_like(ul)*self._timpact[0]
        do= -pwpolyCoeff0-pwpolyCoeff1*(da-pwpolyBreak)
        # Arrays for final coefficients
        c0= pwpolyCoeff0
        c1= 1.+pwpolyCoeff1*self._timpact[0]
        cx= numpy.zeros(len(ul))
        for kk in range(1,len(self._timpact)):
            ul, da, ti, do, c0, c1, cx, \
                pwpolyBreak, pwpolyCoeff0, pwpolyCoeff1=\
                self._update_approx_prevImpact(kk,ul,da,ti,do,c0,c1,cx,
                                               pwpolyBreak,
                                               pwpolyCoeff0,pwpolyCoeff1)
        # Form final c0 by adding cx times ul
        c0-= cx*ul
        return (ul,da,ti,c0,c1)

    def _update_approx_prevImpact(self,kk,ul,da,ti,do,c0,c1,cx,
                                  pwpolyBreak,pwpolyCoeff0,pwpolyCoeff1):
        """Update the lower and upper limits, and the coefficient arrays when
        going through the previous impact"""
        # Compute matrix of upper limits for each current breakpoint and each 
        # previous breakpoint
        da_u= da-(self._timpact[kk]-self._timpact[kk-1])*do
        ti_u= ti+(self._timpact[kk]-self._timpact[kk-1])*c1
        xj= numpy.tile(self._sgapdfs[kk]._kick_interpdOpar_poly.x[::-1],
                       (len(ul),1)).T
        ult= (da_u-xj)/ti_u
        # Determine which of these fall within the previous set of limits,
        # allowing duplicates is easiest
        limitIndx= (ult >= numpy.roll(ul,1))
        limitIndx[:,0]= True
        limitIndx*= (ult <= ul)
        # Only keep those, flatten, add the previous set, and sort; this is the
        # new set of upper limits (barring duplicates, see below)
        ul_u= numpy.append(ult[limitIndx].flatten(),ul)
        # make sure to keep duplicates 2nd (important later when assigning 
        # coefficients to the old limits)
        limitsIndx= numpy.argsort(\
            stats.rankdata(ul_u,method='ordinal').astype('int')-1) 
        limitusIndx= numpy.argsort(limitsIndx) # to un-sort later
        ul_u= ul_u[limitsIndx]
        # Start updating the coefficient arrays
        tixpwpoly= ti*pwpolyCoeff1
        c0_u= c0+tixpwpoly*ul
        cx_u= cx+tixpwpoly
        # Keep other arrays in sync with the limits
        nNewCoeff= len(self._sgapdfs[kk]._kick_interpdOpar_poly.x)
        da_u= numpy.append(numpy.tile(da_u,(nNewCoeff,1))[limitIndx].flatten(),
                            da_u)[limitsIndx]
        ti_u= numpy.append(numpy.tile(ti_u,(nNewCoeff,1))[limitIndx].flatten(),
                           ti_u)[limitsIndx]
        do_u= numpy.append(numpy.tile(do,(nNewCoeff,1))[limitIndx].flatten(),
                           do)[limitsIndx]
        c0_u= numpy.append(numpy.tile(c0_u,(nNewCoeff,1))[limitIndx].flatten(),
                           c0_u)[limitsIndx]
        c1_u= numpy.append(numpy.tile(c1,(nNewCoeff,1))[limitIndx].flatten(),
                           c1)[limitsIndx]
        cx_u= numpy.append(numpy.tile(cx_u,(nNewCoeff,1))[limitIndx].flatten(),
                           cx_u)[limitsIndx]
        # Also update the previous coefficients, figuring out where old limits
        # were inserted
        insertIndx= numpy.sort(\
            limitusIndx[numpy.arange(numpy.sum(limitIndx),len(ul_u))])[:-1]
        pwpolyBreak_u= numpy.append(\
            xj[limitIndx].flatten(),numpy.zeros(len(ul)))[limitsIndx]
        pwpolyCoeff0_u= numpy.append(numpy.tile(\
                numpy.append(\
                    self._sgapdfs[kk]._kick_interpdOpar_poly.c[-1],0.)[::-1],
                (len(ul),1)).T[limitIndx].flatten(),\
                                       numpy.zeros(len(ul)))[limitsIndx]
        pwpolyCoeff1_u= numpy.append(numpy.tile(\
                numpy.append(\
                    self._sgapdfs[kk]._kick_interpdOpar_poly.c[-2],0.)[::-1],
                (len(ul),1)).T[limitIndx].flatten(),\
                                       numpy.zeros(len(ul)))[limitsIndx]
        # Need to do this sequentially, if multiple inserted in one range
        for ii in insertIndx[::-1]:
            pwpolyBreak_u[ii]= pwpolyBreak_u[ii+1]
            pwpolyCoeff0_u[ii]= pwpolyCoeff0_u[ii+1]
            pwpolyCoeff1_u[ii]= pwpolyCoeff1_u[ii+1]
        do_u-= pwpolyCoeff0_u+pwpolyCoeff1_u*(da_u-pwpolyBreak_u)
        # Now update the coefficient arrays
        c0_u+= pwpolyCoeff0_u       
        c1_u+= pwpolyCoeff1_u*ti_u
        # Remove duplicates in limits
        dupIndx= numpy.roll(ul_u,-1)-ul_u != 0
        return (ul_u[dupIndx],da_u[dupIndx],ti_u[dupIndx],do_u[dupIndx],
                c0_u[dupIndx],c1_u[dupIndx],cx_u[dupIndx],
                pwpolyBreak_u[dupIndx],pwpolyCoeff0_u[dupIndx],
                pwpolyCoeff1_u[dupIndx])

    def minOpar(self,dangle,bruteforce=False,_return_raw=False,*args):
        """
        NAME:
           minOpar
        PURPOSE:
           return the approximate minimum parallel frequency at a given angle
        INPUT:
           dangle - parallel angle
           bruteforce= (False) if True, just find the minimum by evaluating where p(Opar,apar) becomes non-zero
        OUTPUT:
           minimum frequency that gets to this parallel angle
        HISTORY:
           2016-01-01 - Written - Bovy (UofT)
           2016-01-02 - Added bruteforce - Bovy (UofT)
        """
        if bruteforce:
            return self._minOpar_bruteforce(dangle)
        if len(args) == 0:
            ul,da,ti,c0,c1= self._approx_pdf(dangle)
        else:
            ul,da,ti,c0,c1= args
        # Find the lower limit of the integration interval
        lowx= ((ul-c0)*(self._tdisrupt-self._timpact[-1])+ul*ti-da)\
            /((self._tdisrupt-self._timpact[-1])*c1+ti)
        nlowx= lowx/(ul-numpy.roll(ul,1))
        nlowx[lowx < 0.]= numpy.inf
        lowbindx= numpy.argmin(nlowx)
        lowbindx= numpy.arange(len(ul))[lowbindx]
        edge= False
        if lowbindx == 0: # edge case
            lowbindx= 1
            lowx[lowbindx]= ul[1]-ul[0]
            edge= True
        if _return_raw:
            return (lowbindx,lowx[lowbindx],edge)
        elif edge:
            return dangle/self._tdisrupt
        else:
            return ul[lowbindx]-lowx[lowbindx]
        
    def _minOpar_bruteforce(self,dangle):
        nzguess= numpy.array([self._meandO,self._meandO])
        sig= numpy.array([numpy.sqrt(self._sortedSigOEig[2]),
                          numpy.sqrt(self._sortedSigOEig[2])])
        while numpy.all(self.pOparapar(nzguess,dangle) == 0.):
            nzguess+= sig/3.
        nzguess= nzguess[nzguess != 0.][0]
        nzval= self.pOparapar(nzguess,dangle)
        guesso= nzguess-numpy.sqrt(self._sortedSigOEig[2])
        while self.pOparapar(guesso,dangle)-10.**-6.*nzval > 0.:
            guesso-= numpy.sqrt(self._sortedSigOEig[2])
        return optimize.brentq(\
            lambda x: self.pOparapar(x*nzguess,dangle)-10.**-6.*nzval,
            guesso/nzguess,1.,xtol=1e-8,rtol=1e-8)*nzguess

    def meanOmega(self,dangle,oned=False,approx=True,tdisrupt=None,norm=True):
        """
        NAME:

           meanOmega

        PURPOSE:

           calculate the mean frequency as a function of angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

           oned= (False) if True, return the 1D offset from the progenitor (along the direction of disruption)

           approx= (True) if True, compute the mean Omega by direct integration of the spline representation

        OUTPUT:

           mean Omega

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if tdisrupt is None: tdisrupt= self._tdisrupt
        if approx:
            dO1D= self._meanOmega_approx(dangle,tdisrupt)
        else:
            if not norm:
                denom= super(streampepperdf,self)._density_par(dangle)
            else:
                denom= self._density_par(dangle,approx=approx)
            dO1D=\
                integrate.quad(lambda T: (T/(1-T*T)\
                                              *numpy.sqrt(self._sortedSigOEig[2])\
                                              +self._meandO)\
                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                   *(1+T*T)/(1-T*T)**2.\
                                   *self.pOparapar(T/(1-T*T)\
                                                       *numpy.sqrt(self._sortedSigOEig[2])\
                                                       +self._meandO,dangle)\
                                   /self._meandO/denom,
                               -1.,1.,
                               limit=100,epsabs=1.e-06,epsrel=1.e-06)[0]\
                               *self._meandO
            if not norm: dO1D*= denom
        if oned: return dO1D
        else:
            return self._progenitor_Omega+dO1D*self._dsigomeanProgDirection\
                *self._sigMeanSign

    def _meanOmega_approx(self,dangle,tdisrupt):
        """Compute the mean frequency as a function of parallel angle using the
        spline representations"""
        ul,da,ti,c0,c1= self._approx_pdf(dangle)
        # Get the density in various forms
        dens_arr= self._density_par_approx(dangle,tdisrupt,True,
                                           ul,da,ti,c0,c1)
        dens= self._density_par_approx(dangle,tdisrupt,False,
                                       ul,da,ti,c0,c1)
        # Compute numerator using the approximate PDF
        # Find the lower limit of the integration interval
        lowbindx,lowx,edge= self.minOpar(dangle,False,True,ul,da,ti,c0,c1)
        ul[lowbindx-1]= ul[lowbindx]-lowx
        # Integrate each interval
        out= numpy.sum(((ul+(self._meandO+c0-ul)/c1)*dens_arr
                        +numpy.sqrt(self._sortedSigOEig[2]/2./numpy.pi)/c1**2.\
                            *(numpy.exp(-0.5*(ul-c0
                                              -c1*(ul-numpy.roll(ul,1))
                                              -self._meandO)**2.
                                         /self._sortedSigOEig[2])
                              -numpy.exp(-0.5*(ul-c0-self._meandO)**2.
                                          /self._sortedSigOEig[2])))\
                           [lowbindx:])
        # Add integration to infinity
        out+= 0.5*(numpy.sqrt(2./numpy.pi)*numpy.sqrt(self._sortedSigOEig[2])\
                       *numpy.exp(-0.5*(self._meandO-ul[-1])**2.\
                                       /self._sortedSigOEig[2])
                   +self._meandO
                   *(1.+special.erf((self._meandO-ul[-1])
                                    /numpy.sqrt(2.*self._sortedSigOEig[2]))))
        # Add integration to edge if edge
        if edge:
            out+= 0.5*(self._meandO*(special.erf(\
                    1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                        *(ul[0]-self._meandO))
                                      -special.erf(\
                    1./numpy.sqrt(2.*self._sortedSigOEig[2])
                    *(dangle/tdisrupt-self._meandO)))
                       +numpy.sqrt(2.*self._sortedSigOEig[2]/numpy.pi)\
                           *(numpy.exp(-0.5*(dangle/tdisrupt-self._meandO)**2.
                                        /self._sortedSigOEig[2])
                              -numpy.exp(-0.5*(ul[0]-self._meandO)**2.
                                          /self._sortedSigOEig[2])))
        return out/dens

################################SAMPLE THE DF##################################
    def _sample_aAt(self,n):
        """Sampling frequencies, angles, and times part of sampling, for stream with gaps"""
        # Use streamdf's _sample_aAt to generate unperturbed frequencies,
        # angles
        Om,angle,dt= super(streampepperdf,self)._sample_aAt(n)
        # Now rewind angles to the first impact, then apply all kicks, 
        # and run forward again
        dangle_at_impact= angle-numpy.tile(self._progenitor_angle.T,(n,1)).T\
            -(Om-numpy.tile(self._progenitor_Omega.T,(n,1)).T)\
            *self._timpact[-1]
        dangle_par_at_impact=\
            numpy.dot(dangle_at_impact.T,
                      self._dsigomeanProgDirection)\
                      *self._sgapdfs[-1]._gap_sigMeanSign
        dOpar= numpy.dot((Om-numpy.tile(self._progenitor_Omega.T,(n,1)).T).T,
                         self._dsigomeanProgDirection)\
                         *self._sgapdfs[-1]._gap_sigMeanSign
        for kk,timpact in enumerate(self._timpact[::-1]):
            # Calculate and apply kicks (points not yet released have 
            # zero kick)
            dOr= self._sgapdfs[-kk-1]._kick_interpdOr(dangle_par_at_impact)
            dOp= self._sgapdfs[-kk-1]._kick_interpdOp(dangle_par_at_impact)
            dOz= self._sgapdfs[-kk-1]._kick_interpdOz(dangle_par_at_impact)
            Om[0,:]+= dOr
            Om[1,:]+= dOp
            Om[2,:]+= dOz
            if kk < len(self._timpact)-1:
                run_to_timpact= self._timpact[::-1][kk+1]
            else:
                run_to_timpact= 0.
            angle[0,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdar(dangle_par_at_impact)\
                +dOr*timpact
            angle[1,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdap(dangle_par_at_impact)\
                +dOp*timpact
            angle[2,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdaz(dangle_par_at_impact)\
                +dOz*timpact
            # Update parallel evolution
            dOpar+=\
                self._sgapdfs[-kk-1]._kick_interpdOpar(dangle_par_at_impact)
            dangle_par_at_impact+= dOpar*(timpact-run_to_timpact)
        return (Om,angle,dt)

