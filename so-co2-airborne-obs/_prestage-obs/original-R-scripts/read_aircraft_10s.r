readac10s=function(){

# R Program to read in HIPPO, ORCAS, and ATom 10-sec merge files, massage time variables, add a strat flag, subset, and write to RData object for use by other programs (aircraft_obspack_merge.r and aircraft_filter_mspo_10s.r)

library('ncdf4')

# specify aircraft data file names
localdatadir='local-data-files'
hippomergefile=paste(localdatadir,'/aircraft-merge-products/HIPPO_all_missions_merge_10s_20121129.tbl',sep='') # this is the official HIPPO merge product
atommergedir=paste(localdatadir,'/aircraft-merge-products',sep='') # these are version 2.0 (21-08-26)
atommergefiles=c('MER10_DC8_ATom-1.nc','MER10_DC8_ATom-2.nc','MER10_DC8_ATom-3.nc','MER10_DC8_ATom-4.nc') 
orcasmergefile=paste(localdatadir,'/aircraft-merge-products/ORCASall.merge10.tbl',sep='') # original release version

# set strat flag cutoffs for use below based upon Jin et al., 2021 (https://doi.org/10.5194/acp-21-217-2021)
stratcoh2o=50
stratcoo3=150
stratcon2o=319

# read in global N2O for detrending aircraft N2O
glbn2ofile=url('ftp://aftp.cmdl.noaa.gov/products/trends/n2o/n2o_annmean_gl.txt')
hlines=61
glbn2o=read.table(glbn2ofile,skip=hlines,header=F,stringsAsFactors=F)
colnames(glbn2o)=c('year','n2o','unc')


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

# add strat flag
hippomerge$strat=rep(0,nrow(hippomerge)) # 0 means trop
h2oref=hippomerge$H2Oppmv_vxl; h2oref[is.na(h2oref)]=hippomerge$H2O_UWV[is.na(h2oref)]
hippomerge$h2oref=h2oref # for output
h2oref[is.na(h2oref)]=0 # if H2O missing treat as if potentially strat
n2oref=hippomerge$N2O_QCLS; n2oref[is.na(n2oref)]=hippomerge$N2O_P[is.na(n2oref)]; n2oref[is.na(n2oref)]=hippomerge$N2O_UGC[is.na(n2oref)]
n2oref=n2oref-(approx(glbn2o$year+0.5,glbn2o$n2o,hippomerge$Year+hippomerge$DOY/365)$y-glbn2o$n2o[glbn2o$year==2009])
hippomerge$n2oref=n2oref # for output
n2oref[is.na(n2oref)]=400 # if N2O missing do not use for filter
o3ref=hippomerge$O3_ppb; o3ref[is.na(o3ref)]=hippomerge$O3_UO3[is.na(o3ref)]
hippomerge$o3ref=o3ref # for output
o3ref[is.na(o3ref)]=0 # if O3 missing do not use for filter
hippomerge$strat[h2oref<stratcoh2o&(o3ref>stratcoo3|n2oref<stratcon2o|(o3ref==0&n2oref==400))]=1 # if either o3 or n2o criteria are met, or if both are missing, consider strat
hippomerge$strat[h2oref==0&o3ref==0&n2oref==400&hippomerge$GGALT<8000]=0 # if all 3 missing assume < 8 km is trop

# select columns and save
colsel=c('Year','Month','Day','Hour','Min','Sec','H.no','flt','n.prof','GGLAT','GGLON','GGALT','PSXC','THETA','CO2.X','CO2_QCLS','CO2_OMS','CO2_AO2','CH4_QCLS','CH4_UGC','CH4_P','SF6_P','SF6_UGC','SF6_CCG','strat','h2oref','n2oref','o3ref')
hippomerge=hippomerge[,is.element(colnames(hippomerge),colsel)]
hippomerge=hippomerge[,match(colsel,names(hippomerge))] # reorder
names(hippomerge)=c('year','mon','day','hour','min','sec','camp','flt','prof','lat','lon','alt','pressure','theta','co2','co2qcls','co2oms','co2ao2','ch4qcls','ch4ucats','ch4panther','sf6panther','sf6ucats','sf6pfp','strat','h2oref','n2oref','o3ref') ## 'co2' = CO2.X
save('hippomerge',file='HIPPO_10s.RData')


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
colsel=c('Year','Month','Day','Hour','Min','Sec','flt','n.prof','GGLAT','GGLON','GGALT','PSXC','THETA','CO2.X','CO2_QCLS','CO2_NOAA','CO2_AO2','CH4_NOAA','CH4_QCLS','strat','h2oref','n2oref')
orcasmerge=orcasmerge[,is.element(colnames(orcasmerge),colsel)]
orcasmerge=orcasmerge[,match(colsel,names(orcasmerge))] # reorder
names(orcasmerge)=c('year','mon','day','hour','min','sec','flt','prof','lat','lon','alt','pressure','theta','co2','co2qcls','co2noaa','co2ao2','ch4noaa','ch4qcls','strat','h2oref','n2oref') ## 'co2' = CO2.X
save('orcasmerge',file='ORCAS_10s.RData')


## read in ATom files, calc strat flag, subset, and output
print('Reading ATom files')

# read and add time variables
atomvar=c('time','UTC_Start','Flight_Date','DLH-H2O/H2O_DLH','UCATS-H2O/H2O_UWV','QCLS-CH4-CO-N2O/N2O_QCLS','GCECD/N2O_PECD','UCATS-GC/N2O_UCATS','NOyO3-O3/O3_CL','UCATS-O3/O3_UCATS','MMS/G_ALT','RF','prof.no','MMS/P','MMS/POT','MMS/G_LAT','MMS/G_LONG','MMS/G_ALT','NOAA-Picarro/CO2_NOAA','QCLS-CO2/CO2_QCLS','AO2/CO2_AO2','CO2.X','NOAA-Picarro/CH4_NOAA','QCLS-CH4-CO-N2O/CH4_QCLS','UCATS-GC/CH4_UCATS','GCECD/CH4_PECD','GCECD/SF6_PECD','UCATS-GC/SF6_UCATS')
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
atommerge$Sec=floor(atommerge$Sec/10)*10 # 'time' in new files is UTC_Mean but time in ObsPack is UTC_Start

# add strat flag
atommerge$strat=rep(0,nrow(atommerge)) # 0 means trop
h2oref=atommerge$H2O_DLH; h2oref[is.na(h2oref)]=atommerge$H2O_UWV[is.na(h2oref)]
atommerge$h2oref=h2oref # for output
h2oref[is.na(h2oref)]=0 # if H2O missing treat as if potentially strat
n2oref=atommerge$N2O_QCLS; n2oref[is.na(n2oref)]=atommerge$N2O_PECD[is.na(n2oref)]; n2oref[is.na(n2oref)]=atommerge$N2O_UCATS[is.na(n2oref)]
n2oref=n2oref-(approx(glbn2o$year+0.5,glbn2o$n2o,atommerge$Year+atomdt$yday/365)$y-glbn2o$n2o[glbn2o$year==2009])
atommerge$n2oref=n2oref # for outptut
### should interpolate PECD for ATom 1
n2oref[is.na(n2oref)]=400 # if N2O missing do not use for filter
o3ref=atommerge$O3_CL; o3ref[is.na(o3ref)]=atommerge$O3_UCATS[is.na(o3ref)]
atommerge$o3ref=o3ref
o3ref[is.na(o3ref)]=0 # if O3 missing do not use for filter
atommerge$strat[h2oref<stratcoh2o&(o3ref>stratcoo3|n2oref<stratcon2o|(o3ref==0&n2oref==400))]=1 # if either o3 or n2o criteria are met, or if both are missing, consider strat
atommerge$strat[h2oref==0&o3ref==0&n2oref==400&atommerge$G_ALT<8000]=0 # if all 3 missing assume < 8 km is trop

# select column and save
colsel=c('Year','Month','Day','Hour','Min','Sec','A.no','RF','prof.no','G_LAT','G_LONG','G_ALT','P','POT','CO2_NOAA','CO2_QCLS','CO2_AO2','CO2.X','CH4_NOAA','CH4_QCLS','CH4_UCATS','CH4_PECD','SF6_PECD','SF6_UCATS','strat','h2oref','n2oref','o3ref')
atommerge=atommerge[,is.element(colnames(atommerge),colsel)]
atommerge=atommerge[,match(colsel,names(atommerge))] # reorder
names(atommerge)=c('year','mon','day','hour','min','sec','camp','flt','prof','lat','lon','alt','pressure','theta','co2','co2qcls','co2ao2','co2x','ch4noaa','ch4qcls','ch4ucats','ch4panther','sf6panther','sf6ucats','strat','h2oref','n2oref','o3ref') ## 'co2' = CO2_NOAA
save('atommerge',file='ATom_10s.RData')


} # end of readac10s function
