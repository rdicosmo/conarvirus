#!/usr/bin/python3

import matplotlib.pyplot as plt
from datetime import date,timedelta
import numpy as np
import csv
import os



regions = {
    'Asia': ['Heilongjiang(China)', 'Henan(China)', 'Hubei(China)', 'India', 'Indonesia', 'Japan', 'Korea, South', 'Malaysia', 'Philippines'],
    'Europe': ['Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom'],
    'Middle East': ['Algeria', 'Egypt', 'Iran', 'Iraq', 'Morocco'],
    'North America': ['British Columbia(Canada)','Ontario(Canada)', 'Dominican Republic', 'US' ],
    'South America': ['Brazil', 'Ecuador']
}    

world = [ x for x in regions ]




#############################
# Loading and filtering data

def reformat_location(location):

    l1,l2=location.split("/")
    if l2=="" or l1==l2:
        return l1
    else:
        return l2+"("+l1+")"


def get_data_from_files():

    files = [ "time_series_covid19_deaths_global.csv", "time_series_covid19_confirmed_global.csv", "time_series_covid19_recovered_global.csv" ] # files where to fetch data

    data2 = dict()
    i=0
    for key in keys:
        data2[key]=dict()
        with open(files[i]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    days = row[4:]
                else:
                    # reformat location
                    location = row[1]+"/"+row[0]
                    p = reformat_location(location)
                    data = [ int(row[j]) for j in range(4,len(row))]
                    # store data
                    data2[key][p] = data                            
                line_count += 1
        i += 1

    #data2["Total"] = dict()
    
    return(data2, days)


def filter_by_dates(data, days, begin, end):

    data2 = dict()
    a = days.index(begin)
    b = days.index(end)
    for k in data:
        data2[k] = dict()
        for p in data[k]:
            data2[k][p] = data[k][p][a:b+1]
            
    return(data2, days[a:b+1])


def filter_by_var_threshold(data, var, threshold, sync):

    data2, xr = dict(), dict()
    for k in data:
        data2[k], xr[k] = dict(), dict()

    for p in data[var]:
        if p in data[var]:
            d = data[var][p]
            if max(d) >= threshold:
                j = list(map(lambda i: i>=threshold, d)).index(True)
                for k in data:
                    if p in data[k]:
                        d2 = data[k][p][j:]
                        if len(d2) >= 2:
                            if sync:
                                data2[k][p] = d2
                                xr[k][p] = range( len(d2) )
                            else:
                                data2[k][p] = data[k][p]
                                xr[k][p] = range( -len(data2[k][p])+1, 1 )
                
    return(data2, xr)


def filter_by_regions(data, l):

    data2 = dict()
    for k in data:
        data2[k] = dict()
        for p in l:
            if p in data[k]:
                data2[k][p] = data[k][p]

    return data2


def add_region(data, name, lst):

    for k in data:
        l = []
        for p in lst:
            if p in data[k]:
                l.append(data[k][p])
        data[k][name] = [sum(x) for x in zip(*l)]

    
#####################################
# Data smoothing and differentiation

def diff(f): # discrete differentiation
    return( [ 0.0 ] + [ f[j]-f[j-1]  for j in range(1, len(f)) ] )

def smooth(y, smooth_parameter, c=3.0 ):  # local linear smoothing
    y2 = y.copy()
    for i in range(smooth_parameter):
        y2 = [ y2[0] ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y2)-1) ] + [ y2[-1] ]
    return y2

def smooth_speed_acceleration(data, smooth_parameter):
    #
    gs, dgs, ddgs = dict(), dict(), dict()
    for key in keys:
        gs[key], dgs[key], ddgs[key] = dict(), dict(), dict()
        for f in data[key]:
            gs[key][f] =  list(map(lambda x: np.exp(x)-1, 
                              smooth(
                                  list(map(lambda x: np.log(1+x), data[key][f])), smooth_parameter   # smoothing of the log(1+x)-transformed function
                              )
            ))
            dgs[key][f] = diff( gs[key][f] )   # speed
            ddgs[key][f] = diff( dgs[key][f] ) # acceleration
            
    return gs,dgs,ddgs


#########################
# Functions to plot data

def style_args(region_list, focus):   # define style parameters for plot and text 

    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta", "red" ]
    lc = len(colors)
    linestyles = [ "-", "--", "-.", ":" ]

    arg_plot, arg_text = dict(), dict()
    k = 0
    for f in region_list:
            
        arg_plot[f], arg_text[f] = dict(), dict()

        arg_plot[f]["color"] = colors[k%lc]
        arg_plot[f]["linestyle"] = linestyles[int(k/lc)]

        arg_text[f]["color"] = colors[k%lc]
        
        
        if f in focus:          # format that depends on focus
            arg_plot[f]["lw"]=2
            arg_text[f]["bbox"] = dict(facecolor='white', alpha=0.5, boxstyle="round", edgecolor=colors[k%lc])
            arg_text[f]["fontsize"] = 10
        else:
            arg_plot[f]["lw"]=1
            arg_text[f]["fontsize"] = 8

        k+=1
            
    return arg_plot, arg_text


def plot(xr, g, gs, dgs, ddgs, arg_plot, arg_text, tmax, size=4):

    lk=len(keys)

    fig = plt.figure(figsize=(size*4,size*lk))
    
    for i in range(lk):
        key = keys[i]
        l = 0
        for (title, fc, yscale, ylim) in [ ("Total number of "+key, gs, ('log', {}), 10.0),
                                                ("Number of "+key+" by day $\\left(\\frac{\\Delta "+key.replace(" ","\\_")+"}{\\Delta t}\\right)$", dgs, ('log', {}), 1.0),
                                                ("Acceleration of the number of "+key+" $\\left(\\frac{\Delta^2 "+key.replace(" ","\\_")+"}{(\Delta t)^2}\\right)$", ddgs, ('symlog',{'linthreshy': 10.0}), 0)
                                                ]: 

            ax = plt.subplot( lk, 3, 3*i + l+1 )
            plt.title(title)

            for f in fc[key]:
                
                #if l==0:
                #    lw, color = arg_plot[f]["lw"], arg_plot[f]["color"]  # same color and width as graph
                #    plt.plot( xr[key][f], g[key][f], "+", mew=lw/2.0, ms=lw*2.5, color=color )

                if len(xr[key][f]) > l:
                    plt.plot( xr[key][f][l:], fc[key][f][l:], **arg_plot[f] )  
                    if fc[key][f][-1] > ylim or yscale[0]!="log":
                        plt.text( xr[key][f][-1] , fc[key][f][-1], f, **arg_text[f] )

            plt.yscale(yscale[0],**yscale[1])
            if ylim!=0:
                plt.ylim(ylim)
                
            x1,_ = ax.get_xlim()
            x2 = x1 + tmax - l
            plt.xlim(x1,x2+(x2-x1)/5.0)

            plt.grid(True,which="both")
            l+=1
            
    plt.tight_layout()    
    plt.subplots_adjust(top=0.92)

    return(fig)


#########################################
#### Main function


# Common part

def prepare_data(begin="", end=""):
    
    data, days = get_data_from_files()
    
    if begin=="":
        begin = days[0]
    if end=="":
        end = days[-1]
    
    data, days = filter_by_dates(data, days, begin, end) 
    data, xr = filter_by_var_threshold(data, "deaths", 10, False)
    
    return data, days, xr 




def curves(sync=False, smooth_parameter=15, size=4):

    data, days, xr = prepare_data()

    ### Top-10-confirmed countries
    
    d = data["deaths"]
    c = [x for x in d]
    mv = np.array([ max(d[x]) for x in d ])
    bests = list( mv.argsort()[-10:][::-1] )
    region_list = [ c[i] for i in bests ]
    focus = []
    
    data2, xr = filter_by_var_threshold(data, 'deaths', 10, sync=True)
    data2 = filter_by_regions(data2, region_list)
    tmax = max([ len(data2['deaths'][x]) for x in data2['deaths'] ] )
    arg_plot, arg_text = style_args(region_list, focus)
    
    day = days[-1]
    print("Generating graphs for deaths-Top-10 countries, day="+day+", sync=",sync)
    print(region_list)
    gs, dgs, ddgs = smooth_speed_acceleration(data2, smooth_parameter)
    fig = plot(xr, data2, gs, dgs, ddgs, arg_plot, arg_text, tmax, size=size)
    plt.suptitle("Dynamics of Covid-19 for the 10 countries with maximal number of deaths (day 0: first day such that $deaths \\geq 10$), "+str(day))

    filename = "top10"
    if sync:
        filename += "_sync"
    plt.savefig(filename+".png", dpi=dpi)
    plt.savefig(filename+".pdf")


    # Global graph + graphs by continents

    # add continents
    for x in regions:
        add_region(data, x, regions[x])

    # add world   
    add_region(data, 'World', world)

    
    
    for x in [x for x in regions]+['World']:
        
        if x=='World':
            region_list = [x for x in regions]
            sync = False
            tit  = "Dynamics of Covid-19 in all continents, day 0="+str(day)
        else:
            region_list = [y for y in regions[x]]
            sync = True
            tit = "Dynamics of Covid-19 in "+x+" (day 0: first day such that $deaths \\geq 10$), "+str(day)

        data2,xr = filter_by_var_threshold(data, 'deaths', 10, sync)
        data2 = filter_by_regions(data2, region_list)
        tmax = max( [ len(data2['deaths'][x]) for x in data2['deaths'] ] )
        arg_plot, arg_text = style_args(region_list, [])
    
        day = days[-1]
        print("Generating graphs for "+x+", day="+day+", sync=",sync)
        gs, dgs, ddgs = smooth_speed_acceleration(data2, smooth_parameter)
        fig = plot(xr, data2, gs, dgs, ddgs, arg_plot, arg_text, tmax, size=size)
        plt.suptitle(tit)

        filename = x.replace(" ","_")
        if sync:
            filename += "_sync"
        plt.savefig(filename+".png", dpi=dpi)
        plt.savefig(filename+".pdf")

    
    
        

def regularise_anim(sync=False,focus=[],smooth_parameter=15, begin="", end="",size=4):

    data, days, xr, arg_plot, arg_text = prepare_data(sync, begin, end)
    day = days[-1]
    print("Graphs for several smoothing parameters, sync=",sync)
    
    smooth_parameters = range(smooth_parameter + 1)

    if sync:
        fic="smooth_sync"
    else:
        fic="smooth"
    ns = len(smooth_parameters)
    for n in range(ns):
        print("Smooth parameter:",smooth_parameters[n])
        gs, dgs, ddgs = smooth_speed_acceleration(data, n)
        fig = plot(xr, data, gs, dgs, ddgs, arg_plot, arg_text, tmax, size=size)
        plt.suptitle(str(day)+", smooth="+str(n))
        plt.savefig("fig/"+fic+"_%02d.png"%n, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 15 -loop 0 "
        
    lns = list(range( ns )) + [ns-1]*4 + list(range(ns-1,-1,-1)) + [0]*4
    src = " ".join([ "fig/"+fic+"_%02d.png"%i for i in lns ])
    print("Generate animations from : "+src)
    os.system( anim_command+src+" "+fic+".mp4" )
    os.system( anim_command+src+" "+fic+".gif" )


    
def evolution_anim(sync=False,begin="",end="", size=4):

    print("Graphs for several dates, sync=",sync)

    data, days, xr, arg_plot, arg_text = prepare_data(sync, begin, end)
    
    smooth_parameter = 15
    gs, dgs, ddgs = smooth_speed_acceleration(data, smooth_parameter)
    
    filename="evolution"
    if sync:
        filename="evolution_sync"
        
    for t in range(0,tmax):
        
        print("t=",t,"day=",days[t])

        # Crop all data 
        data2, xr2, gs2, dgs2, ddgs2 = dict(), dict(), dict(), dict(), dict() 
        for key in keys:
            data2[key], xr2[key], gs2[key], dgs2[key], ddgs2[key] = dict(), dict(), dict(), dict(), dict()
            for f in data[key]:

                z = max(0, len(data[key][f]) - tmax + t + 1)
                data2[key][f] = data[key][f][0:z]
                xr2[key][f] = xr[key][f][0:z]
                gs2[key][f] = gs[key][f][0:z]
                dgs2[key][f] = dgs[key][f][0:z]
                ddgs2[key][f] = ddgs[key][f][0:z]
                
        fig = plot(xr2, data2, gs2, dgs2, ddgs2, arg_plot, arg_text, tmax, size=size)
        plt.suptitle(str(days[t]))
        plt.savefig("fig/"+filename+"_%02d.png"%t, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 30 -loop 0 "
        
    ltmax = list( range( 3,tmax-1 ) ) 
    src = " ".join([ "fig/"+filename+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".mp4" )
    os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".gif" )


#########################
# Main global parameters

keys = [ "deaths", "confirmed cases", "recovered cases" ] # data to show

dpi = 100 # graph quality (for png/mp4)

curves(sync=True)

exit(1)

regularise_anim(sync=True)
regularise_anim(sync=False, begin=begin, end=end)

evolution_anim(sync=True)
evolution_anim(sync=False, begin=begin, end=end)

