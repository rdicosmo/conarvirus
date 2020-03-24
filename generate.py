#!/usr/bin/python3

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from datetime import date,timedelta
import numpy as np
import csv
import os



keys = [ "deaths", "confirmed"]#, "recovered" ] # data to show


# pour charger les données à partir des fichiers csv


def get_data_from_files(death_threshold = 10):

    files = [ "time_series_covid19_deaths_global.csv", "time_series_covid19_confirmed_global.csv" ]
        
    data2 = dict()

    i=0

    for key in keys:
        data2[key]=dict()
        with open(files[i]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    day = row[-1]
                else:
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
        
    return(data2, day)


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
    return y2
    

def smooth2(y, n):
    y2 = y.copy()
    ly2 = len(y2)
    y2 = [ np.mean( y2[ max(  0,i-n) : min(ly2-1,i+n)  ] ) for i in range(ly2) ]
    return(y2)
    


def trace(d, sm=0, t=-1, log=True, sync=False, size=5 ):

    tmax = len(d["deaths"]["Hubei(China)"])  # max duration
    if t==-1:
        t = tmax
        
    lk = len(keys)
    
    day =  ( date(2020, 1, 21) + timedelta(days=t) ) 
    
    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta" ]
    lc = len(colors)
    linestyles = [ "-", "--", "-.", ":" ]
    
    fig = plt.figure(figsize=(size*lk, size*2))
    fig.suptitle(day)

    shift = 5

    i=0
    for key in keys:

        g, gs, dg, dgs, ddgs = dict(), dict(), dict(), dict(), dict()

        for f in d[key]:
            
            z = len(d[key][f])-tmax+t
            if z>=2:
                g[f] = d[key][f]     
                dg[f] = [ 0.0 ] + [ g[f][j]-g[f][j-1]  for j in range(1, len(g[f])) ]
                dgs[f] =  smooth( dg[f], sm)  # one smooths dg
                gs[f] = [ g[f][0] ]           # one then computes smoothed-g by intergrating smoothed-dg
                for j in  range(1, len(g[f])):
                    gs[f].append(gs[f][j-1] + dgs[f][j])
                ddgs[f] = np.gradient( dgs[f] )

        ax1 = plt.subplot(3, lk, i+1)
        plt.title( "Total number of "+keys[i] )
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if not sync:
                xr = range(-z+1,1)
            else:
                xr = range(0,z)
            if z >= 2:
                plt.plot( xr, g[f][0:z], "+", color=colors[k%lc] )
                plt.plot( xr, gs[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)] )
                if gs[f][z-1]>10.0:
                    plt.text( xr[-1] , gs[f][z-1], f, color=colors[k%lc], fontsize=6 )
            k+=1
        if log:
            plt.yscale('log')
            plt.ylim(bottom=10.0)
        if not sync:
            plt.xlim(-z+1,shift)
        else:
            plt.xlim(0,t+shift)
        plt.grid(True,which="both")
            
        ax2 = plt.subplot(3, lk, lk + i+1)
        plt.title(keys[i]+" by day $\\left(\\frac{\Delta "+keys[i]+"}{\Delta t}\\right)$")
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if not sync:
                xr = range(-z+1,1)
            else:
                xr = range(0,z)
            if z >= 2:
                plt.plot(xr, dg[f][0:z], "+", color=colors[k%lc])
                plt.plot(xr, dgs[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)] )
                if dgs[f][z-1]>=1.0:
                    plt.text( xr[-1], dgs[f][z-1], f, color=colors[k%lc], fontsize=6 )
            k+=1
        plt.ylim(bottom=1.0)
        if log:
            plt.yscale('log')
        if not sync:
            plt.xlim(-z+1,shift)
        else:
            plt.xlim(0,t+shift)
        plt.grid(True,which="both")
        
        
        ax3 = plt.subplot(3, lk, 2*lk + i+1)
        plt.title("Acceleration of "+keys[i]+" $\\left(\\frac{\Delta^2 "+keys[i]+"}{\Delta t^2}\\right)$")
        plt.plot([0.0]*(t+1), "--", color="grey")
        k=0

        plt.grid(True,which="both")
        #ys = list(range(-10,0))+list(range(1,11))+list(range(10,110,10)) 
        if not sync:
            plt.xlim(-z+1,shift)
            #for y in ys:
            #    plt.hlines(y, -t+1,shift+1, "lightgrey" )
        else:
            plt.xlim(0,t+shift)
            #for y in ys:
            #    plt.hlines(y, 0,t+shift+1, "lightgrey" )

        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if not sync:
                xr = range(-z+1,1)
            else:
                xr = range(0,z)
            if z >= 2:
                # accélération
                plt.plot( xr, ddgs[f][0:z], label=f, color=colors[k%lc], linestyle=linestyles[int(k/lc)])
                plt.text( xr[-1], ddgs[f][z-1], f, color=colors[k%lc], fontsize=6 )
            k+=1
        if log:
            plt.yscale('symlog')
        
        
            
        i=i+1

    #plt.subplots_adjust(right=0.5)
    plt.tight_layout()
    #handles, labels = ax.get_legend_handles_labels()
    #plt.figlegend(handles, labels, borderaxespad=0.0, loc='upper center', ncol=7)
    
    plt.subplots_adjust(top=0.92)



anim_command = "convert -verbose -scale 80% -delay 10 -loop 0 "
    

def regularise(sync=False):

    print("Graphs for several smoothing parameters, sync=",sync)
    
    ns = 15
    sm = range(ns)

    d,_ = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)
    
    fic="smooth"
    if sync:
        fic="smooth_sync"
    for n in range(ns):
        print(sm[n])
        trace(d, sm[n], sync=sync) 
        plt.savefig("fig/"+fic+"_%02d.png"%n)
        plt.close('all')

    lns = [0]*4 + list(range( ns )) + [ns-1]*4 + list(range(ns-1,-1,-1))
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in lns ])
    print("Generate animations from : "+src)
    os.system( anim_command+src+" "+fic+".gif" )
    os.system( anim_command+src+" "+fic+".mp4" )


def evolution(sync=False):
    
    print("Graphs for several dates, sync=",sync)
    
    sm = 15
    
    d,_ = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)

    tmax = len(d["deaths"]["Hubei(China)"])  # max duration
    
    fic="evolution"
    if sync:
        fic="evolution_sync"
    for t in range(3,tmax+1):
        print(t)
        trace(d, sm, t, sync=sync)
        plt.savefig("fig/"+fic+"_%02d.png"%t)
        plt.close('all')

    ltmax = list( range( 3,tmax+1 ) ) + [tmax]*50
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    os.system( anim_command+src+" "+fic+".gif" )
    os.system( anim_command+src+" "+fic+".mp4" )



def curve(sync=False):
    
    d,day = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)

    print("Curve for day "+day)
    
    trace(d, 15, sync=sync, size=8)
    plt.suptitle("Day 0 = "+day)
    
    plt.savefig(day.replace("/","_")+".pdf")


    
#### 


regularise()
regularise(True)

evolution()
evolution(True)


curve()
