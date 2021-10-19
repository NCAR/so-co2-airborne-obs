acfiltmspopfp=function(readnew=F,filt=T){

# R Program to filter aircraft data for strong local continental influences, subtract off NOAA in situ SPO, and write out flat text files

library('ncdf4')
localdatadir='local-data-files'

# read in aircraft data
if(readnew){
	source('read_aircraft_pfp.r')
	readacpfp()
} else {
	# read in preprocessed aircraft files from read_aircraft_pfp.r
	load('HIPPO_PFP.RData')
	load('ATom_PFP.RData')
}


# calculate datetime variables
hippodt=ISOdatetime(hippomerge$year,hippomerge$mon,hippomerge$day,hippomerge$hour,hippomerge$min,hippomerge$sec,tz='UTC')
atomdt=ISOdatetime(atommerge$year,atommerge$mon,atommerge$day,atommerge$hour,atommerge$min,atommerge$sec,tz='UTC')


# read in NOAA in situ record from SPO
sponc=nc_open(paste(localdatadir,'/obspack_co2_1_GLOBALVIEWplus_v6.0_2020-09-11/data/nc/co2_spo_surface-insitu_1_allvalid.nc',sep=''))
spoco2=data.frame(cbind(ncvar_get(sponc,'time_decimal'),t(ncvar_get(sponc,'time_components')),ncvar_get(sponc,'value')*1E6)) ; colnames(spoco2)=c('date','year','mon','day','hour','min','sec','co2')
qcflag=ncvar_get(sponc,'qcflag'); spoco2$co2[substr(qcflag,1,1)!='.']=NA; spoco2$co2[substr(qcflag,2,2)!='.']=NA
spodt=ISOdatetime(spoco2$year,spoco2$mon,spoco2$day,spoco2$hour,spoco2$min,spoco2$sec,tz='UTC')


# HIPPO
print('Processing HIPPO file')

# filter
if(filt){
	ints=read.table(paste(localdatadir,'/hippo_xsect_filt_datetime.txt',sep=''),header=T) 
	startdt=ISOdatetime(ints$startyear,ints$startmon,ints$startday,ints$starthour,ints$startmin,ints$startsec,tz='UTC')
	stopdt=ISOdatetime(ints$stopyear,ints$stopmon,ints$stopday,ints$stophour,ints$stopmin,ints$stopsec,tz='UTC')
	blfilt=rep(T,nrow(hippomerge))
	for(i in c(1:nrow(ints))){
		blfilt[difftime(hippodt,startdt[i])>=0&difftime(hippodt,stopdt[i])<=0]=F
	}
	print(paste('Filtering ',sum(!blfilt),' of ',length(blfilt),' HIPPO obs (',round(sum(!blfilt)/length(blfilt)*100,1),'%)',sep=''))
	hippodt=hippodt[blfilt]
	hippomerge=hippomerge[blfilt,]
}

# calculate differences
hippomerge$co2mspo=round(hippomerge$co2-approx(as.POSIXct(spodt),spoco2$co2,as.POSIXct(hippodt))$y,3) ## co2 = 'CO2_CCG'
hippomerge$co2mqcls=round(hippomerge$co2-hippomerge$co2qcls,3)
hippomerge$co2moms=round(hippomerge$co2-hippomerge$co2oms,3)
hippomerge$co2mao2=round(hippomerge$co2-hippomerge$co2ao2,3)
hippomerge$ch4mqcls=round(hippomerge$ch4pfp-hippomerge$ch4qcls,3)

# write out
write(names(hippomerge),'HIPPO_SO_mSPO_pfp.txt',ncol=ncol(hippomerge))
write(t(hippomerge),'HIPPO_SO_mSPO_pfp.txt',ncol=ncol(hippomerge),append=T)

print(apply(!is.na(hippomerge),2,sum))


# ATom
print('Processing ATom file')

# filter
if(filt){
        ints=read.table(paste(localdatadir,'/atom_xsect_filt_datetime.txt',sep=''),header=T)
        startdt=ISOdatetime(ints$startyear,ints$startmon,ints$startday,ints$starthour,ints$startmin,ints$startsec,tz='UTC')
        stopdt=ISOdatetime(ints$stopyear,ints$stopmon,ints$stopday,ints$stophour,ints$stopmin,ints$stopsec,tz='UTC')
        blfilt=rep(T,nrow(atommerge))
        for(i in c(1:nrow(ints))){
                blfilt[difftime(atomdt,startdt[i])>=0&difftime(atomdt,stopdt[i])<=0]=F
        }
        print(paste('Filtering ',sum(!blfilt),' of ',length(blfilt),' ATom obs (',round(sum(!blfilt)/length(blfilt)*100,1),'%)',sep=''))
        atomdt=atomdt[blfilt]
        atommerge=atommerge[blfilt,]
}

# calculate differences
atommerge$co2mspo=round(atommerge$co2-approx(as.POSIXct(spodt),spoco2$co2,as.POSIXct(atomdt))$y,2) ## co2 = 'CO2_PFP'
atommerge$co2mqcls=round(atommerge$co2-atommerge$co2qcls,3)
atommerge$co2mao2=round(atommerge$co2-atommerge$co2ao2,3)
atommerge$co2mnoaa=round(atommerge$co2-atommerge$co2noaa,3)
atommerge$ch4mqcls=round(atommerge$ch4pfp-atommerge$ch4qcls,3)
atommerge$ch4mnoaa=round(atommerge$ch4pfp-atommerge$ch4noaa,3)

# write out
write(names(atommerge),'ATOM_SO_mSPO_pfp.txt',ncol=ncol(atommerge))
write(t(atommerge),'ATOM_SO_mSPO_pfp.txt',ncol=ncol(atommerge),append=T)


} # end of acfiltmspopfp function
