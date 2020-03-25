#!/usr/bin/python3

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from datetime import date,timedelta
import numpy as np
import csv
import os



keys = [ "deaths", "confirmed"] # data to show


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


def diff(f): # discrete differential
    return( [ 0.0 ] + [ f[j]-f[j-1]  for j in range(1, len(f)) ] )

    
def integ(f, a=0): # discrete integral
    g = [ a ]
    for j in range(1, len(f)):
        g.append( g[j-1]+f[j] )
    return(g)
        

def smooth(y, n, c=3.0):
    y2 = y.copy()
    for i in range(n):
        #y2 = [ ( y2[0]*(c-1) + y2[1] )/c ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y)-1) ] + [ (y2[len(y)-1]*(c-1) + y2[len(y)-2])/c ]
        y2 = [ y2[0] ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y)-1) ] + [ y2[-1] ]
    return y2
    


def trace(d, sm=0, t=-1, log=True, sync=False, size=4.5 ):

    lw=1.5
    
    tmax = len(d["deaths"]["Hubei(China)"])  # max duration
    if t==-1:
        t = tmax
        
    lk = len(keys)
    
    day =  ( date(2020, 1, 21) + timedelta(days=t) ) 
    
    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta" ]
    lc = len(colors)
    linestyles = [ "-", "--", "-.", ":" ]
    
    fig = plt.figure(figsize=(size*4,size*lk))
    fig.suptitle(str(day)+" (smooth=%d)"%sm)

    shift = 5

    i=0
    for key in keys:

        g, gs, dg, dgs, ddg, ddgs = dict(), dict(), dict(), dict(), dict(), dict()

        for f in d[key]:
            
            z = len(d[key][f])-tmax+t
            if z>=2:
                
                g[f] = d[key][f]     
                dg[f] = diff( g[f] )
                
                ddg[f] = diff( dg[f] )

                gs[f] = smooth( g[f], sm )
                dgs[f] = smooth( dg[f], sm )
                ddgs[f] = smooth( ddg[f], sm )
                
        ax1 = plt.subplot(lk, 3, 3*i + 1)
        plt.title( "Total number of "+keys[i] )
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if not sync:
                xr = range(-z+1,1)
            else:
                xr = range(0,z)
            if z >= 2:
                plt.plot( xr, g[f][0:z], "+", color=colors[k%lc], mew=lw/2., ms=lw*1.5 )
                plt.plot( xr, gs[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)], lw=lw )
                if gs[f][z-1] > 10.0:
                    plt.text( xr[-1] , gs[f][z-1], f, color=colors[k%lc], fontsize=7 )
            k+=1
        if log:
            plt.yscale('log')
            plt.ylim(bottom=10.0)
        if not sync:
            plt.xlim(-z+1,shift)
        else:
            plt.xlim(0,t+shift)
        plt.grid(True,which="both")
            
        ax2 = plt.subplot(lk, 3, 3*i + 2)
        plt.title(keys[i]+" by day $\\left(\\frac{\Delta "+keys[i]+"}{\Delta t}\\right)$")
        k=0
        for f in d[key]:
            z = len(d[key][f])-tmax+t
            if not sync:
                xr = range(-z+1,1)
            else:
                xr = range(0,z)
            if z >= 2:
                plt.plot( xr, dg[f][0:z], "+", color=colors[k%lc], mew=lw/2., ms=lw*1.5)
                plt.plot( xr, dgs[f][0:z], color=colors[k%lc], linestyle=linestyles[int(k/lc)], lw=lw )
                if dgs[f][z-1]>=1.0:
                    plt.text( xr[-1], dgs[f][z-1], f, color=colors[k%lc], fontsize=7 )
            k+=1
        plt.ylim(bottom=1.0)
        if log:
            plt.yscale('log')
        if not sync:
            plt.xlim(-z+1,shift)
        else:
            plt.xlim(0,t+shift)
        plt.grid(True,which="both")
        
        
        ax3 = plt.subplot(lk, 3, 3*i + 3)
        plt.title("Acceleration of "+keys[i]+" $\\left(\\frac{\Delta^2 "+keys[i]+"}{(\Delta t)^2}\\right)$")
        plt.plot([0.0]*(t+1), "--", color="grey", lw=lw)
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
                plt.plot( xr, ddg[f][0:z], "+", color=colors[k%lc], mew=lw/2., ms=lw*1.5)
                plt.plot( xr, ddgs[f][0:z], label=f, color=colors[k%lc], linestyle=linestyles[int(k/lc)], lw=lw)
                plt.text( xr[-1], ddgs[f][z-1], f, color=colors[k%lc], fontsize=7 )
            k+=1
        if log:
            plt.yscale('symlog')
                    
        i=i+1

    #plt.subplots_adjust(right=0.5)
    plt.tight_layout()
    #handles, labels = ax.get_legend_handles_labels()
    #plt.figlegend(handles, labels, borderaxespad=0.0, loc='upper center', ncol=7)
    
    plt.subplots_adjust(top=0.92)



    
def regularise(sync=False):

    print("Graphs for several smoothing parameters, sync=",sync)
    
    ns = 16
    sm = range(ns)

    d,_ = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)
    
    fic="smooth"
    if sync:
        fic="smooth_sync"
    for n in range(ns):
        print(sm[n])
        trace(d, sm[n], sync=sync) 
        plt.savefig("fig/"+fic+"_%02d.png"%n, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 50 -loop 0 "
        
    lns = [0]*4 + list(range( ns )) + [ns-1]*4 + list(range(ns-1,-1,-1))
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in lns ])
    print("Generate animations from : "+src)
    os.system( anim_command+src+" "+fic+".mp4" )
    os.system( anim_command+src+" "+fic+".gif" )


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
        plt.savefig("fig/"+fic+"_%02d.png"%t, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 10 -loop 0 "
        
    ltmax = list( range( 3,tmax ) ) 
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    os.system( anim_command+src+" -delay 300 fig/"+fic+"_%02d.png"%(tmax)+" "+fic+".mp4" )
    os.system( anim_command+src+" -delay 300 fig/"+fic+"_%02d.png"%(tmax)+" "+fic+".gif" )



def curve(sync=False, sm=15):
    
    d,day = get_data_from_files()
    d = filter_by_var(d, "deaths", 10, sync=sync)

    print("Curve for day "+day)
    
    trace(d, sm, sync=sync)
    plt.suptitle("Day 0 = "+str(day)+" (smooth=%d)"%sm )
    
    plt.savefig(str(day).replace("/","_")+".pdf", dpi=dpi)

    
#### 

dpi = 130

#curve()

#regularise()
#regularise(True)

evolution()
evolution(True)

