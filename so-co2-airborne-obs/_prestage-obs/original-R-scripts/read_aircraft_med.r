readacmed=function(){

# R Program to read in HIPPO, ORCAS, and ATom MED merge files, massage time variables, add a strat flag, subset, and write to RData object for use by aircraft_filter_mspo_med.r

library('ncdf4')

# specify aircraft data file names
localdatadir='local-data-files'
hippomergefile=paste(localdatadir,'/aircraft-merge-products/HIPPO_medusa_flasks_merge_insitu_20121129.tbl',sep='') # this is the official HIPPO merge product
atommergedir=paste(localdatadir,'/aircraft-merge-products',sep='') # these are version 2.0 (21-08-26)
atommergefiles=c('MER-MED_DC8_ATom-1.nc','MER-MED_DC8_ATom-2.nc','MER-MED_DC8_ATom-3.nc','MER-MED_DC8_ATom-4.nc')  # these files have problems (see below)
orcasmergefile=paste(localdatadir,'/aircraft-merge-products/ORCASall.mergeMED.tbl',sep='') # original release version

# set strat flag cutoffs for use below based upon Jin et al., 2021 (https://doi.org/10.5194/acp-21-217-2021)
stratcoh2o=50
stratcoo3=150
stratcon2o=319

# read in global N2O for detrending aircraft N2O
glbn2ofile=url('ftp://aftp.cmdl.noaa.gov/products/trends/n2o/n2o_annmean_gl.txt')
hlines=61
glbn2o=read.table(glbn2ofile,skip=hlines,header=F,stringsAsFactors=F)
colnames(glbn2o)=c('year','n2o','unc')

# for HIPPO, need to interpolate prof variable from 10-sec merge - load from read_aircraft_10s.r output before reading PFP data below
if(file.exists('HIPPO_10s.RData')){
      load('HIPPO_10s.RData')
} else {
        source('read_aircraft10s.r')
        readac10s()
}
hippomerge10s=hippomerge # 'hippomerge' reused below
hippo10sdt=ISOdatetime(hippomerge$year,hippomerge$mon,hippomerge$day,hippomerge$hour,hippomerge$min,hippomerge$sec,tz='UTC')


## read in HIPPO file, calc strat flag, subset, and output
print('Reading HIPPO file')

# read and add time variables
hippomerge=read.table(hippomergefile,header=T)
hippodt=strptime(paste(hippomerge[,"Year"],hippomerge[,"DOY"]),format='%Y %j',tz='UTC')+hippomerge[,"UTC"] # DOY is day of year of takeoff; UTC is seconds since midnight on day of takeoff
hippomerge$Month=as.POSIXlt(hippodt)$mon+1
hippomerge$Day=as.POSIXlt(hippodt)$mday
hippomerge$Hour=as.POSIXlt(hippodt)$hour
hippomerge$Min=as.POSIXlt(hippodt)$min
hippomerge$Sec=as.POSIXlt(hippodt)$sec

# filter in situ records for overlap
hippomerge$CO2_QCLS[hippomerge$wt.qcls<0.5]=NA
hippomerge$CO2_AO2[hippomerge$wt.ao2<0.5]=NA
hippomerge$CO2_OMS[hippomerge$wt.oms<0.5]=NA

# interpolate prof from 10 sec file
hippomerge$prof=approx(as.POSIXct(hippo10sdt),hippomerge10s$prof,as.POSIXct(hippodt),method='constant',f=0)$y

# add strat flag
hippomerge$strat=rep(0,nrow(hippomerge)) # 0 means trop
h2oref=hippomerge$H2Oppmv_vxl; h2oref[is.na(h2oref)]=hippomerge$H2O_UWV[is.na(h2oref)]
hippomerge$h2oref=h2oref # for output
h2oref[is.na(h2oref)]=0 # if H2O missing treat as if potentially strat
n2oref=hippomerge$N2O_QCLS 
n2oref=n2oref-(approx(glbn2o$year+0.5,glbn2o$n2o,hippomerge$Year+hippomerge$DOY/365)$y-glbn2o$n2o[glbn2o$year==2009])
hippomerge$n2oref=n2oref # for output
n2oref[is.na(n2oref)]=400 # if N2O missing do not use for filter
o3ref=hippomerge$O3_ppb; o3ref[is.na(o3ref)]=hippomerge$O3_UO3[is.na(o3ref)]
hippomerge$o3ref=o3ref # for output
o3ref[is.na(o3ref)]=0 # if O3 missing do not use for filter
hippomerge$strat[h2oref<stratcoh2o&(o3ref>stratcoo3|n2oref<stratcon2o|(o3ref==0&n2oref==400))]=1 # if either o3 or n2o criteria are met, or if both are missing, consider strat
hippomerge$strat[h2oref==0&o3ref==0&n2oref==400&hippomerge$GGALT<8000]=0 # if all 3 missing assume < 8 km is trop

# select columns and save
colsel=c('Year','Month','Day','Hour','Min','Sec','H.no','flt','prof','GGLAT','GGLON','GGALT','PSXC','THETA','CO2_MED','CO2_QCLS','CO2_OMS','CO2_AO2','strat','h2oref','n2oref','o3ref')
hippomerge=hippomerge[,is.element(colnames(hippomerge),colsel)]
hippomerge=hippomerge[,match(colsel,names(hippomerge))] # reorder
names(hippomerge)=c('year','mon','day','hour','min','sec','camp','flt','prof','lat','lon','alt','pressure','theta','co2','co2qcls','co2oms','co2ao2','strat','h2oref','n2oref','o3ref') ## 'co2' = CO2_MED
save('hippomerge',file='HIPPO_MED.RData')


## read in ORCAS file, calc strat flag, subset, and output
print('Reading ORCAS file')

# read and add time variables
orcasmerge=read.table(orcasmergefile,header=T)
orcasdt=strptime(paste(orcasmerge[,"Year"],orcasmerge[,"DOY"]),format='%Y %j',tz='UTC')+orcasmerge[,"UTC"] # DOY is day of year of takeoff; UTC is seconds since midnight on day of takeoff
orcasmerge$Month=as.POSIXlt(orcasdt)$mon+1
orcasmerge$Day=as.POSIXlt(orcasdt)$mday
orcasmerge$Hour=as.POSIXlt(orcasdt)$hour
orcasmerge$Min=as.POSIXlt(orcasdt)$min
orcasmerge$Sec=as.POSIXlt(orcasdt)$sec

# filter in situ records for overlap ### add for HIPPO and ATom?
orcasmerge$CO2_NOAA[orcasmerge$wt.CO2_NOAA<0.5]=NA
orcasmerge$CO2_QCLS[orcasmerge$wt.CO2_QCLS<0.5]=NA
orcasmerge$CO2_AO2[orcasmerge$wt.CO2_AO2<0.5]=NA
orcasmerge$CO2.X[orcasmerge$wt.CO2.X<0.5]=NA

# add strat flag
orcasmerge$strat=rep(0,nrow(orcasmerge)) # 0 means trop
h2oref=orcasmerge$VMR_VXL; h2oref[is.na(h2oref)]=orcasmerge$H2O_NOAA[is.na(h2oref)]
orcasmerge$h2oref=h2oref # for output
h2oref[is.na(h2oref)]=0 # if H2O missing treat as if potentially strat
n2oref=orcasmerge$N2O_QCLS
n2oref=n2oref-(approx(glbn2o$year+0.5,glbn2o$n2o,orcasmerge$Year+orcasmerge$DOY/365)$y-glbn2o$n2o[glbn2o$year==2009])
orcasmerge$n2oref=n2oref # for output
n2oref[is.na(n2oref)]=400 # if N2O missing do not use for filter
orcasmerge$strat[h2oref<stratcoh2o&(n2oref<stratcon2o|n2oref==400)]=1 # no O3, if n2o criteria is met or missing, consider strat
orcasmerge$strat[h2oref==0&n2oref==400&orcasmerge$G_ALT<8000]=0 # if both missing assume < 8 km is trop

# select columns and save
colsel=c('Year','Month','Day','Hour','Min','Sec','flt','n.prof','GGLAT','GGLON','GGALT','PSXC','THETA','CO2_MED','CO2.X','CO2_QCLS','CO2_NOAA','CO2_AO2','strat','h2oref','n2oref')
orcasmerge=orcasmerge[,is.element(colnames(orcasmerge),colsel)]
orcasmerge=orcasmerge[,match(colsel,names(orcasmerge))] # reorder
names(orcasmerge)=c('year','mon','day','hour','min','sec','flt','prof','lat','lon','alt','pressure','theta','co2','co2x','co2qcls','co2noaa','co2ao2','strat','h2oref','n2oref') ## 'co2' = CO2_MED
save('orcasmerge',file='ORCAS_MED.RData')


## read in ATom files, calc strat flag, subset, and output
print('Reading ATom files')

# read and add time variables
atomvar=c('time','Flight_Date','DLH-H2O/H2O_ppmv','UCATS-H2O/H2O_UWV','QCLS-CH4-CO-N2O/N2O_QCLS','NOyO3-O3/O3_CL','UCATS-O3/O3_UO3','MMS/G_ALT','RF','prof.no','MMS/P','MMS/POT','MMS/G_LAT','MMS/G_LONG','MMS/G_ALT','MEDUSA/CO2_MED','NOAA-Picarro/CO2_NOAA','QCLS-CO2/CO2_QCLS','AO2/CO2_AO2','NOAA-Picarro/CH4_NOAA','QCLS-CH4-CO-N2O/CH4_QCLS')
atommerge=NULL
for(i in c(1:4)){
	atomnc=nc_open(paste(atommergedir,'/',atommergefiles[i],sep=''))
       	count=length(ncvar_get(atomnc,'time'))
	campdata=NULL
	for(var in atomvar){
		if(i==1&var=='UCATS-H2O/H2O_UWV'){ # no UCATS H2O on ATom-1
			campdata=cbind(campdata,rep(NA,count))
		} else {
			campdata=cbind(campdata,ncvar_get(atomnc,var))
		}
	}
	campdata=cbind(campdata,rep(i,count)) # A.no
       	nc_close(atomnc)
	atommerge=rbind(atommerge,campdata)
}
atommerge=data.frame(atommerge,stringsAsFactors=F)
names(atommerge)=c(gsub('.*/','',atomvar),'A.no')
atommerge$YYYYMMDD=atommerge$Flight_Date
atomdt=as.POSIXlt(ISOdatetime(2016,1,1,0,0,0,tz='UTC')+atommerge$time,tz='UTC')
atommerge$Year=atomdt$year+1900
atommerge$Month=as.POSIXlt(atomdt)$mon+1
atommerge$Day=as.POSIXlt(atomdt)$mday
atommerge$Hour=as.POSIXlt(atomdt)$hour
atommerge$Min=as.POSIXlt(atomdt)$min
atommerge$Sec=as.POSIXlt(atomdt)$sec

# filter in situ records for overlap
atommerge$CO2_NOAA[atommerge$wt.CO2_NOAA<0.5]=NA
atommerge$CO2_QCLS[atommerge$wt.CO2_QCLS<0.5]=NA
atommerge$CO2_AO2[atommerge$wt.CO2_AO2<0.5]=NA

# add strat flag
atommerge$strat=rep(0,nrow(atommerge)) # 0 means trop
h2oref=atommerge$H2O_ppmv; h2oref[is.na(h2oref)]=atommerge$H2O_UWV[is.na(h2oref)]
atommerge$h2oref=h2oref # for output
h2oref[is.na(h2oref)]=0 # if H2O missing treat as if potentially strat
n2oref=atommerge$N2O_QCLS
n2oref=n2oref-(approx(glbn2o$year+0.5,glbn2o$n2o,atommerge$Year+atomdt$yday/365)$y-glbn2o$n2o[glbn2o$year==2009])
atommerge$n2oref=n2oref # for outptut
n2oref[is.na(n2oref)]=400 # if N2O missing do not use for filter
o3ref=atommerge$O3_CL; o3ref[is.na(o3ref)]=atommerge$O3_UO3[is.na(o3ref)]
atommerge$o3ref=o3ref
o3ref[is.na(o3ref)]=0 # if O3 missing do not use for filter
atommerge$strat[h2oref<stratcoh2o&(o3ref>stratcoo3|n2oref<stratcon2o|(o3ref==0&n2oref==400))]=1 # if either o3 or n2o criteria are met, or if both are missing, consider strat
atommerge$strat[h2oref==0&o3ref==0&n2oref==400&atommerge$G_ALT<8000]=0 # if all 3 missing assume < 8 km is trop

# select column and save
colsel=c('Year','Month','Day','Hour','Min','Sec','A.no','RF','prof.no','G_LAT','G_LONG','G_ALT','P','POT','CO2_MED','CO2_NOAA','CO2_QCLS','CO2_AO2','CH4_NOAA','CH4_QCLS','strat','h2oref','n2oref','o3ref')
atommerge=atommerge[,is.element(colnames(atommerge),colsel)]
atommerge=atommerge[,match(colsel,names(atommerge))] # reorder
names(atommerge)=c('year','mon','day','hour','min','sec','camp','flt','prof','lat','lon','alt','pressure','theta','co2','co2noaa','co2qcls','co2ao2','ch4noaa','ch4qcls','strat','h2oref','n2oref','o3ref') ## 'co2' = CO2_MED
save('atommerge',file='ATom_MED.RData')


} # end of readacmed function
