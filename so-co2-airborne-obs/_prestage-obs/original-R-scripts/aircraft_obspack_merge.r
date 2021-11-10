acobspackmrg=function(readnew=F){

# R program to merge flight parameters from campaign 10-sec merge files onto obspack IDs for use by read_aircraft_models.r

# Notes:
# merging is needed because model files generally do not include pressure, theta, campaign, flight, and profile variables
# also models either do not report or round time and position values, such that merging on those not possible, so using 2 steps:
# 1) merge 10-sec flight data and NOAA Obpack here and write out
# 2) in read_aircraft_models.r merge model output with these output files
# because the model output corresponds to many different ObsPack versions - need to download each and match flight data for each
# CT2017, CT2019B, CTE2020, MIROC, TM5pCO2, and CAMS have obspack_id variables (CAMS only has obspack_id and co2)
# CarboScope text files do not include obspack_id, so need to merge based on row matching

# read in aircraft data
if(readnew){
        source('read_aircraft_10s.r')
        readac10s()
} else {
        # read in preprocessed aircraft files from read_aircraft_10s.r
        load('HIPPO_10s.RData')
        load('ORCAS_10s.RData')
        load('ATom_10s.RData')
}


# point to ObsPack directories and define short names
localdatadir='local-data-files'
ncdirs=c(
paste(localdatadir,'/obspack_co2_1_GLOBALVIEWplus_v4.2.2_2019-06-05/data/nc',sep=''),
paste(localdatadir,'/obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12/data/nc',sep=''), # CT2019B includes this
paste(localdatadir,'/obspack_co2_1_GLOBALVIEWplus_v6.0_2020-09-11/data/nc',sep=''),
paste(localdatadir,'/obspack_co2_1_ATom_v4.0_2020-04-06/data/nc',sep=''),
paste(localdatadir,'/obspack_co2_1_CARBONTRACKER_CT2017_2018-05-02/data/nc',sep='')
# CT2017 includes these 3: obspack_co2_1_ORCAS_v2.0_2017-04-05, obspack_co2_1_GLOBALVIEWplus_v3.1_2017-10-18, obspack_co2_1_NRT_v4.0_2017-09-08 (includes ATom-2 but fluxes only go through 12/2016 so not included in paper)
)
ops=c('GLOBALVIEWplus_v4.2.2','GLOBALVIEWplus_v5.0','GLOBALVIEWplus_v6.0','ATom_v4.0','CARBONTRACKER_CT2017')


# specify file names and campaigns
ncfiles=c( 'co2_hip_aircraft-insitu_59_allvalid.nc', 'co2_orc_aircraft-insitu_3_allvalid-merge10.nc', 'co2_tom_aircraft-insitu_1_allvalid.nc' )
camps=c('HIPPO','ORCAS','ATOM')


# loop on ObsPack, merge, and write out
for(i in c(1:length(ncdirs))){
	ncdir=ncdirs[i]
	op=ops[i]
	# loop on campaign
	for(j in c(1:3)){
		camp=camps[j]
		ncfile=ncfiles[j]
		if(!grepl('ATom',ncdir)|camp=='ATOM'){ # only process ATom v4.0 ObsPack for ATom
			print(paste(op,camp))
			ncin=nc_open(paste(ncdir,'/',ncfile,sep=''))
			ncdat=data.frame(cbind(t(ncvar_get(ncin,'time_components')),ncvar_get(ncin,'obspack_id')),stringsAsFactors=F) ; colnames(ncdat)=c('year','mon','day','hour','min','sec','obspack_id')
print(head(ncdat))
			mergefile=get(paste(tolower(camp),'merge',sep='')) # from .RData load above
			print(dim(mergefile))
			print(dim(ncdat))
			mrgdat=merge(mergefile,ncdat,by=c('year','mon','day','hour','min','sec'))
			print(dim(mrgdat))
			if(camp=='ATOM'&!grepl('CT2017',ncdir)){ # ATom (except for CT2017) fails to match 7925 because ObsPack includes test flights that are not in the 10s merge product. Only report days:
				print(unique(paste(ncdat$year,ncdat$mon,ncdat$day)[!is.element(paste(ncdat$year,ncdat$mon,ncdat$day,ncdat$hour,ncdat$min,ncdat$sec),paste(mrgdat$year,mrgdat$mon,mrgdat$day,mrgdat$hour,mrgdat$min,mrgdat$sec))]))
			} else { # HIPPO fails to match 6 values because of uneven seconds in the ObsPack, ATom fails to match 11 values in CT2017
				print(paste(ncdat$year,ncdat$mon,ncdat$day,ncdat$hour,ncdat$min,ncdat$sec)[!is.element(paste(ncdat$year,ncdat$mon,ncdat$day,ncdat$hour,ncdat$min,ncdat$sec),paste(mrgdat$year,mrgdat$mon,mrgdat$day,mrgdat$hour,mrgdat$min,mrgdat$sec))])
			}
			if(camp=='ORCAS'){ # no 'camp' variable
				mrgdat=mrgdat[,c('year','mon','day','hour','min','sec','flt','prof','pressure','theta','strat','obspack_id')]
				write(c('year month day hour min sec flt prof pressure theta strat obspack_id'),paste(camp,'_obspack_',op,'_merge.txt',sep=''))
			} else {
				mrgdat=mrgdat[,c('year','mon','day','hour','min','sec','camp','flt','prof','pressure','theta','strat','obspack_id')]
				write(c('year month day hour min sec camp flt prof pressure theta strat obspack_id'),paste(camp,'_obspack_',op,'_merge.txt',sep=''))
			}
			write(t(mrgdat),paste(camp,'_obspack_',op,'_merge.txt',sep=''),ncol=ncol(mrgdat),append=T)
		}
	}
}


} # end of acobspackmrg function
