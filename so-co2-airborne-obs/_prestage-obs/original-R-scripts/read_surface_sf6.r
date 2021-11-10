# R program to read in and monthly-average Southern Ocean NOAA station SF6 data

## could remove all seasonal and climatological aggregation and plotting code since only monthly file used

readsurfsf6=function(){

library('ncdf4')

# General options:
endyear=2020
localdatadir='local-data-files'

# Aggregation options:
minnmon=2 # lowest number of months allowed for seasonal average
meanwin=c(1998.9,2020.2) # window for calculating means, inclusive

# Plotting options
zoomyr=1999 
ylm=c(-0.10,0.15) 

# Select which sites to use:
stationlist=read.table('SO_SF6_stationlist.txt',header=T,stringsAsFactors=F) # use = 0 for do not use, 2 for SO, 1 was for reference but no longer used - use refcol below 
print(stationlist)


# Read in surface data:

# set up arrays
allsta=data.frame(cbind(rep(seq(1957,endyear),each=12),rep(seq(1,12),times=endyear-1957+1)))
colnames(allsta)=c('year','month')
yrfrac=allsta$year+(allsta$mon-0.5)/12
monseas=rep(1,nrow(allsta)) # DJF
monseas[allsta$month>2&allsta$month<6]=2 # MAM
monseas[allsta$month>5&allsta$month<9]=3 # JJA
monseas[allsta$month>8&allsta$month<12]=4 # SON
seasyear=trunc(yrfrac+1/12) # shift Dec to next year
allseas=data.frame(cbind(aggregate(monseas,by=list(seas=monseas,year=seasyear),mean)$x,aggregate(yrfrac,by=list(seas=monseas,year=seasyear),mean)$x))
colnames(allseas)=c('seas','yrfrac')
alllat=NULL
alllon=NULL
allalt=NULL

# loop over records
for(i in c(1:nrow(stationlist))){

	stasf6=NULL
	sta=stationlist$station[i]
	lab=stationlist$lab[i]
	meth=stationlist$method[i]
	use=stationlist$use[i]
	labcode=stationlist$labcode[i]
	print(c(sta,lab,meth,use))

	infile=url(paste('ftp://ftp.cmdl.noaa.gov/data/trace_gases/sf6/flask/surface/sf6_',tolower(sta),'_surface-flask_1_ccgg_month.txt',sep=''))
	line1=readLines(infile,n=1) ; nhead=as.numeric(tail(unlist(strsplit(line1,' ')),1))
	names=readLines(infile,n=nhead)[nhead] ; names=unlist(strsplit(names,' ')) ; names=names[3:length(names)]
	indata=read.table(infile,skip=nhead,stringsAsFactors=F); colnames(indata)=names
	stasf6=data.frame(cbind(indata$year,indata$month,indata$value)); colnames(stasf6)=c('year','month','sf6')
	stasf6$sf6[stasf6$sf6==-999.99]=NA

	# aggregate records by season
	if(!is.null(stasf6)){
		oldnames=colnames(allsta)
		allsta=merge(allsta,stasf6,by=c('year','month'),all.x=T)
		colnames(allsta)=c(oldnames,paste(sta,'_',lab,'_',meth,sep=''))
		# aggregate by season
		seassf6=aggregate(allsta[,ncol(allsta)],by=list(seas=monseas,year=seasyear),mean,na.rm=T)$x # with na.rm=T so returns value even if only 1 month present (alldiffs filtered for < 2 months below)
		oldnames=colnames(allseas)
		allseas=cbind(allseas,seassf6) # since aggregating after merge, all rows present
		colnames(allseas)=c(oldnames,paste(sta,'_',lab,'_',meth,sep=''))
		alllat=c(alllat,stationlist$lat[i])
		alllon=c(alllon,stationlist$lon[i])
		allalt=c(allalt,stationlist$masl[i])
	}

} # loop on surface record

# print out time periods 
for(i in c(3:ncol(allsta))){
	ind=c(1:nrow(allsta))[!is.na(allsta[,i])]
	print(paste(colnames(allsta)[i],',',month.abb[allsta$month[ind[1]]],allsta$year[ind[1]],'-',month.abb[allsta$month[tail(ind,1)]],allsta$year[tail(ind,1)]))
}

# write out results
write(paste(names(allsta),collapse=' '),'SO_SF6_monthly.txt')
write(t(allsta),'SO_SF6_monthly.txt',append=T,ncol=ncol(allsta))
#write('record lat lon alt','SO_SF6_locations.txt')
#write(rbind(names(allsta)[3:ncol(allsta)],alllat,alllon,allalt),'SO_SF6_locations.txt',append=T,ncol=4)
allloc=read.table('SO_SF6_locations.txt',header=T,stringsAsFactors=F)


#### code from here to end not really needed as only SO_SF6_monthly.txt used


# Calc diffs for all stations using NOAA in situ SPO record as the reference
alldiffsmon=allsta # monthly resolution
refcol=which(names(allsta)=='SPO_NOAA_flask')
for(i in c(3:ncol(allsta))){
	alldiffsmon[,i]=allsta[,i]-allsta[,refcol]
}
alldiffsmonclim=aggregate(alldiffsmon[alldiffsmon$year>=meanwin[1]&alldiffsmon$year<=meanwin[2],3:ncol(alldiffsmon)],by=list(month=alldiffsmon$month[alldiffsmon$year>=meanwin[1]&alldiffsmon$year<=meanwin[2]]),mean,na.rm=T) # allows missing months
alldiffsmonclimcomp=apply(alldiffsmonclim[,which(stationlist$use==2)+1],1,mean,na.rm=T)

alldiffs=data.frame(cbind(aggregate(monseas,by=list(seas=monseas,year=seasyear),mean)$x,aggregate(yrfrac,by=list(seas=monseas,year=seasyear),mean)$x,aggregate(alldiffsmon[,3:ncol(alldiffsmon)],by=list(seas=monseas,year=seasyear),mean,na.rm=T)[,3:ncol(alldiffsmon)]))
allnmon=data.frame(cbind(aggregate(monseas,by=list(seas=monseas,year=seasyear),mean)$x,aggregate(yrfrac,by=list(seas=monseas,year=seasyear),mean)$x,aggregate(!is.na(alldiffsmon[,3:ncol(alldiffsmon)]),by=list(seas=monseas,year=seasyear),sum)[,3:ncol(alldiffsmon)]))
alldiffs[,3:ncol(alldiffs)][allnmon[,3:ncol(alldiffs)]<minnmon]=NA ## allow seasons with one missing month, but not two
colnames(alldiffs)[1:2]=c('seas','yrfrac')

# Calc long term mean and sd of diffs, from seasonal differences
meandiff=apply(alldiffs[alldiffs$yrfrac>=meanwin[1]&alldiffs$yrfrac<=meanwin[2],],2,mean,na.rm=T)
sddiff=apply(alldiffs[alldiffs$yrfrac>=meanwin[1]&alldiffs$yrfrac<=meanwin[2],],2,sd,na.rm=T)
sumdiff=apply(!is.na(alldiffs[alldiffs$yrfrac>=meanwin[1]&alldiffs$yrfrac<=meanwin[2],]),2,sum,na.rm=T)
meandiffseas=NULL
sddiffseas=NULL
for(seas in c(1:4)){
	meandiffseas=rbind(meandiffseas,apply(alldiffs[alldiffs$yrfrac>=meanwin[1]&alldiffs$yrfrac<=meanwin[2]&alldiffs$seas==seas,],2,mean,na.rm=T))
	sddiffseas=rbind(sddiffseas,apply(alldiffs[alldiffs$yrfrac>=meanwin[1]&alldiffs$yrfrac<=meanwin[2]&alldiffs$seas==seas,],2,sd,na.rm=T))
}
meandiffseas=data.frame(meandiffseas)
sddiffseas=data.frame(sddiffseas)
sddiffseas$seas=meandiffseas$seas

# Make plots:
library('RColorBrewer')
cols=rep(brewer.pal(12,'Paired'),5) # can handle up to 60 stations
pchs=rep(21,length(cols)) # NOAA
numsta=ncol(allsta)-2
bgs=cols # these are fill colors for pch 21-25
cols[which(stationlist$method=='underway')]=rgb(0,0,0) # these are edge colors for pch 21-25

seasname=c('DJF','MAM','JJA','SON')
for(seas in c(1,3)){

# lat grad of diffs

png(paste('so_station_sf6diff_gradient_',meanwin[1],'-',meanwin[2],'_',seasname[seas],'.png',sep=''),height=1200,width=1200,pointsize=30)
par(mar=c(5,5,4,2)+0.1)

plot(as.numeric(stationlist$lat),meandiffseas[meandiffseas$seas==seas,3:ncol(meandiffseas)],type='n',xlim=c(-90,max(as.numeric(stationlist$lat))),ylim=ylm,main=substitute(paste('Southern Ocean Seasonal ',SF[6],' Gradient - ',v),list(v=seasname[seas])),ylab=expression(paste(Delta,SF[6],' (ppt)')),xlab='Latitude (degrees N)',cex.main=1.2,cex.axis=1.2,cex.lab=1.2)
abline(h=0)
mtext(paste(stationlist$station[refcol-2],' ',stationlist$lab[refcol-2],' ',stationlist$method[refcol-2],' Subtracted, ',meanwin[1],'-',meanwin[2],' Average and SD',sep=''),3,0.3)
segwd=0.2
stasel=c(1:numsta)[c(1:numsta)+2!=refcol&!is.na(meandiffseas[meandiffseas$seas==seas,3:(length(meandiff))])]
y=c(0,as.numeric(meandiffseas[meandiffseas$seas==seas,3:length(meandiff)][stasel]))
x=c(-90,stationlist$lat[stasel])
w=c(10000,as.numeric(1/sddiffseas[sddiffseas$seas==seas,3:length(meandiff)][stasel]^2)) # 10000 for SPO equiv of SD of 0.01
new=data.frame(x=seq(-90,ceiling(max(x)),1))
lines(new$x,predict.lm(lm(y ~ poly(x,3),weights=w),new),lwd=2)
for(i in stasel){
	segments(stationlist$lat[i],meandiffseas[meandiffseas$seas==seas,i+2]-sddiffseas[sddiffseas$seas==seas,i+2],stationlist$lat[i],meandiffseas[meandiffseas$seas==seas,i+2]+sddiffseas[sddiffseas$seas==seas,i+2],col=cols[i])
	segments(stationlist$lat[i]-segwd,meandiffseas[meandiffseas$seas==seas,i+2]-sddiffseas[sddiffseas$seas==seas,i+2],stationlist$lat[i]+segwd,meandiffseas[meandiffseas$seas==seas,i+2]-sddiffseas[sddiffseas$seas==seas,i+2],col=cols[i])
	segments(stationlist$lat[i]-segwd,meandiffseas[meandiffseas$seas==seas,i+2]+sddiffseas[sddiffseas$seas==seas,i+2],stationlist$lat[i]+segwd,meandiffseas[meandiffseas$seas==seas,i+2]+sddiffseas[sddiffseas$seas==seas,i+2],col=cols[i])
        points(stationlist$lat[i],meandiffseas[meandiffseas$seas==seas,i+2],pch=pchs[i],bg=bgs[i],col=cols[i],cex=1.5,lwd=2)
	if(any(!is.na(meandiffseas[meandiffseas$seas==seas,i+2]))) text(stationlist$lat[i],ylm[1],stationlist$sta[i],col=cols[i],srt=90,offset=0,adj=c(0,0.5))
}
labsel=stationlist$lab[stasel]; pchsel=pchs[stasel]; pchsel=pchsel[!duplicated(labsel)]; labsel=labsel[!duplicated(labsel)]
legend('topleft',labsel,pch=pchsel,pt.bg='black',cex=1.0,col='gray30',ncol=2)
if(any(stationlist$method=='underway')) legend('topright','Underway data points outlined in black',cex=0.75,bty='n')

dev.off()

} # loop on season


# Plot composite seasonal cycle

compflag=paste(tolower(substr(stationlist$station[stationlist$use==2],1,1)),collapse='')
refflag='s'

png(paste('so_station_composite_sf6diff_seascycle_',compflag,'-',refflag,'_',meanwin[1],'-',meanwin[2],'.png',sep=''),height=1200,width=1800,pointsize=30)
par(mar=c(5,5,4,2)+0.1)

plot(seq(0.5,11.5),alldiffsmonclim[c(7:12,1:6),3],type='n',xlim=c(0,12),ylim=ylm,main=expression(paste('Southern Ocean Climatological Monthly Mean ',SF[6])),ylab=expression(paste(Delta,SF[6],' (ppt)')),xlab='Month',cex.main=1.2,cex.lab=1.2,axes=F)
box()
axis(2,cex.axis=1.2)
axis(1,at=c(0:12),labels=F,cex.axis=1.5)
axis(1,seq(0.5,11.5),labels=c('J','A','S','O','N','D','J','F','M','A','M','J'),cex.axis=1.3,tick=F)
mtext(paste(stationlist$station[refcol-2],' ',stationlist$lab[refcol-2],' ',stationlist$method[refcol-2],' Subtracted, ',meanwin[1],'-',meanwin[2],sep=''),3,0.3)
abline(h=0)
for(i in c(1:numsta)[stationlist$use==2]){
        points(seq(0.5,11.5),alldiffsmonclim[c(7:12,1:6),i+1],type='b',pch=pchs[i],bg=bgs[i],col=cols[i],cex=1.5,lwd=2)
}
lines(seq(0.5,11.5),alldiffsmonclimcomp[c(7:12,1:6)],col='grey10',lwd=10)

legend('topleft',names(alldiffsmonclim)[c(1:numsta)[stationlist$use==2]+1],col=cols[c(1:numsta)[stationlist$use==2]],pch=pchs[c(1:numsta)[stationlist$use==2]],pt.bg=bgs[c(1:numsta)[stationlist$use==2]],cex=0.75,pt.cex=0.75,lwd=2,ncol=3) # pt.lwd?

dev.off()

} # end of readsurfsf6()
