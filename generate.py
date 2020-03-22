#!/usr/bin/python3

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from datetime import date,timedelta
import numpy as np
import csv
import os



keys = [ "deaths", "confirmed"] #, "recovered" ] # data to show


# pour charger les données à partir des fichiers csv

def get_data_from_files(death_threshold = 10):

    files = [ "time_series_19-covid-Deaths.csv", "time_series_19-covid-Confirmed.csv", "time_series_19-covid-Recovered.csv" ]
        
    data2 = dict()

    i=0

    for key in keys:
        data2[key]=dict()
        with open(files[i]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    # reformat location
                    location = row[1]+"/"+row[0]
                    data = [ int(row[j]) for j in range(4,len(row))]
                    l1,l2=location.split("/")
                    if l2=="" or l1==l2:
                        p = l1
                    else:
                        p = l2+"("+l1+")"
                    # store data
                    data2[key][p] = data
                            
                line_count += 1
        i += 1
        
    return(data2)


def filter_by_var(data, var, threshold, sync=False):

    data2=dict()
    for k in data:
        data2[k]=dict()
        for p in data[k]:
            d = data[var][p]
            if max(d) >= threshold:
                j = list(map(lambda i: i>=threshold, d)).index(True)
                d2 = data[k][p][j:]
                if len(d2) >= 2:
                    if sync:
                        data2[k][p] = d2
                    else:
                        data2[k][p] = data[k][p]

    return(data2)


def smooth(y, n):

    y2 = y.copy()
    for i in range(n):
        y2 = [ y2[0] ] + [ (y2[i-1]+y2[i]+y2[i+1])/3.0 for i in range(1,len(y2)-1) ] + [ y2[-1] ]
#        y2 = [ (y2[0]+y2[1])/2.0 ] + [ (y2[i-1]+y2[i]+y2[i+1])/3.0 for i in range(1,len(y2)-1) ] + [ (y2[-2]+y2[-1])/2.0 ]
    return y2
    


def trace(d, sm=0, t=-1, log=True ):

    tmax = len(d["deaths"]["Hubei(China)"])  # max duration
    if t==-1:
        t = tmax
        
    lk = len(keys)
    
    jour = ( date(2020, 1, 21) + timedelta(days=t) ).isoformat()
    print(jour)
    
    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta","blue","red","green" ]*5
    lc = len(colors)
    linestyles = [ "-", "--", "-.", ":" ]
    
    fig = plt.figure(figsize=(5*lk, 10))
    fig.suptitle(jour)

    if log:
        myplot=plt.semilogy
    else:
        myplot=plt.plot

    i=0
    for key in keys:

        g, dg, dgs, ddgs = dict(), dict(), dict(), dict()

        for f in d[key]:
            
            z = len(d[key][f])-tmax+t
            if z>=2:
                g[f] = d[key][f]     
                dg[f] = [ 0.0 ] + [ g[f][j]-g[f][j-1]  for j in range(1, len(g[f])) ]
                dgs[f] =  smooth( dg[f], sm)
                ddgs[f] = np.gradient( dgs[f] )

        ax1 = plt.subplot(3, lk, i+1)
        plt.title( keys[i] )
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if z >= 2:
                myplot(g[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)] )
                plt.text( z-1, g[f][z-1], f, color=colors[k%lc], fontsize=6 )
            k+=1
        plt.ylim(bottom=10.0)
        plt.xlim((0,t+5))
        
        ax2 = plt.subplot(3, lk, lk + i+1)
        plt.title("Speed ("+keys[i]+" by day)")
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if z >= 2:
                myplot(dg[f][0:z], "+", color=colors[k%lc])
                myplot(dgs[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)] )
                if dgs[f][z-1]>=1.0:
                    plt.text( z-1, dgs[f][z-1], f, color=colors[k%lc], fontsize=6 )
            k+=1
        plt.ylim(bottom=1.0)
        plt.xlim((0,t+5))
        
        ax3 = plt.subplot(3, lk, 2*lk + i+1)
        plt.title("Acceleration ($\Delta$["+keys[i]+" by day)]")
        plt.plot([0.0]*(t+1), "--", color="grey")
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if z >= 2:
                # accélération
                myplot( ddgs[f][0:z], label=f, color=colors[k%lc], linestyle=linestyles[int(k/lc)])
                if ddgs[f][z-1]>=1.0:
                    plt.text( z-1, ddgs[f][z-1], f, color=colors[k%lc], fontsize=6)
            k+=1        
        plt.ylim(bottom=1.0)
        plt.xlim((0,t+5))
            #if i==0:
        #    plt.ylim(bottom=-20.0)
        #else:
        #    plt.ylim(bottom=-1200.0)

        i=i+1
        
    plt.tight_layout()
    #handles, labels = ax.get_legend_handles_labels()
    #plt.figlegend(handles, labels, borderaxespad=0.0, loc='upper center', ncol=7)
    
    plt.subplots_adjust(top=0.92)


def regularise(sync=False):
    
    ns = 15
    sm = range(ns)

    d = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)
    print(d)
    
    fic="smooth"
    if sync:
        fic="smooth_sync"
    for n in range(ns):
        print(sm[n])
        trace(d, sm[n]) 
        plt.savefig("fig/"+fic+"_%02d.png"%n)
        plt.close('all')

    lns = [0]*4 + list(range( ns )) + [ns-1]*4 + list(range(ns-1,-1,-1))
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in lns ])
    print("Generate animations from : "+src)
    os.system( "convert -verbose -delay 10 -loop 0 "+src+" "+fic+".gif" )
    os.system( "convert -verbose -delay 10 -loop 0 "+src+" "+fic+".mp4" )


def evolution(sync=False):

    sm = 15
    
    d = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)

    tmax = len(d["deaths"]["Hubei(China)"])  # max duration
    
    fic="evolution"
    if sync:
        fic="evolution_sync"
    for t in range(3,tmax+1):
        print(t)
        trace(d, sm, t)
        plt.savefig("fig/"+fic+"_%02d.png"%t)
        plt.close('all')

    ltmax = list( range( 3,tmax+1 ) ) + [tmax]*50
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    os.system( "convert -verbose -delay 10 -loop 0 "+src+" "+fic+".gif" )
    os.system( "convert -verbose -delay 10 -loop 0 "+src+" "+fic+".mp4" )

    
#### 


#regularise()
regularise(True)

#evolution()
evolution(True)

