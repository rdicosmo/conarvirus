#!/usr/bin/python3

import matplotlib.pyplot as plt
from datetime import date,timedelta,datetime
import numpy as np
import csv
import os


keys = [ "deaths", "confirmed cases", "recovered cases" ] # data to process
death_threshold = 10


regions = {
    'Asia': ['Heilongjiang(China)', 'Henan(China)', 'Hubei(China)', 'India', 'Indonesia', 'Japan', 'Korea, South', 'Malaysia', 'Philippines', 'Beijing(China)', 'Guangdong(China)', 'Shandong(China)', 'Iran', 'Iraq', 'Lebanon', 'Pakistan', 'Bangladesh', 'Anhui(China)', 'Chongqing(China)', 'Hainan(China)', 'Hebei(China)', 'Shanghai(China)', 'Thailand'],
    'Europe': ['Albania', 'Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'Diamond Princess(Boat)', 'Czechia', 'Finland', 'Luxembourg', 'Slovenia', 'Iceland', 'Serbia', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Lithuania', 'Ukraine'],
    'North America': ['British Columbia(Canada)','Ontario(Canada)', 'Dominican Republic', 'US', 'Quebec(Canada)', 'Panama', 'Peru' ],
    'South America': ['Argentina', 'Brazil', 'Ecuador', 'Mexico', 'Chile', 'Colombia'],
    'Australia': ['New South Wales(Australia)'],
    'Africa': ['Burkina Faso', 'Tunisia', 'Algeria', 'Egypt', 'Israel', 'Congo (Kinshasa)', 'Ghana', 'Morocco'],
    'World': ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Australia']
}    


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
                    location = row[1]+"/"+row[0]
                    p = reformat_location(location)
                    if p=="Diamond Princess":
                        p="Diamond Princess(Boat)"
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


def filter_by_var_threshold(data, days, var, threshold=death_threshold, sync=False):

    data2, xr = dict(), dict()
    for k in data:
        data2[k], xr[k] = dict(), dict()

    for p in data[var]:
        if p in data[var]:
            d = data[var][p]
            #print(p,d)
            if max(d) >= threshold:
                j = list(map(lambda i: i>=threshold, d)).index(True)
                if j<len(d)-1:
                    for k in data:
                        if p in data[k]:
                            d2 = data[k][p][j:]
                            if sync:
                                data2[k][p] = d2
                                xr[k][p] = range( len(d2) )
                            else:
                                data2[k][p] = data[k][p]
                                xr[k][p] = range( -len(data2[k][p])+1, 1 )#[datetime.strptime(days[-1],'%m/%d/%y')+timedelta(d) for d in range( -len(data2[k][p])+1, 1 )]

    return(data2, xr)


def filter_by_regions(data, l, bool=True):

    data2 = dict()
    for k in data:
        data2[k] = dict()
        for p in data[k]:
            if (p in l)==bool:
                data2[k][p] = data[k][p]

    return data2


def add_regions(data):

    for name in regions:
        lst=regions[name]
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

def smooth(y, sm, c=3.0 ):  # local linear smoothing
    y2 = y.copy()
    for i in range(sm):
        y2 = [ y2[0] ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y2)-1) ] + [ y2[-1] ]
    return y2

def compute_fcts(data, sm):
    #
    dg, gs, dgs, ddgs, gr = dict(), dict(), dict(), dict(), dict()
    for key in keys:

        # calcul des dérivées (avec régularisation)
        dg[key], gs[key], dgs[key], ddgs[key], gr[key] = dict(), dict(), dict(), dict(), dict()
        
        for f in data[key]:
            dg[key][f] = diff( data[key][f] )
            
            gs[key][f] =  list(map(lambda x: np.exp(x)-1, 
                              smooth(
                                  list(map(lambda x: np.log(1+x), data[key][f])), sm   # smoothing of the log(1+x)-transformed function
                              )
            ))
            dgs[key][f] = diff( gs[key][f] )   # speed
            ddgs[key][f] = diff( dgs[key][f] ) # acceleration


            if key=='confirmed cases':
                th=300
            else:
                th=30
            gr[key][f]=[ np.nan ]
            for i in range(1,len(gs[key][f])):
                if gs[key][f][i-1] < th:
                    gr[key][f].append( np.nan )
                else:
                    gr[key][f].append( gs[key][f][i]/gs[key][f][i-1]-1.0 )
                            
    return [data, dg, gs, dgs, ddgs, gr]


#########################
# Functions to plot data

def style_args(data, region_list):   # define style parameters for plot and text for each region/country

    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta", "red", "blue", "green", "magenta" ]
    lc = len(colors)
    linestyles = [ "-", "--", ":", "-." ]

    # sort countries by max number of deaths
    
    arg_plot = dict()
    k = 0

    for phase in [0,1]: # phase 0: regions, phase 1: sub-regions, countries
        for f in region_list:
            if (phase==0 and f in regions) or (phase==1 and f not in regions):
                arg_plot[f] = dict()
                arg_plot[f]["color"] = colors[k%lc]
                arg_plot[f]["linestyle"] = linestyles[int(k/lc)]
                k+=1
                
    return arg_plot


def plot(region_list, what_to_plot, focus, xr, fcts, arg_plot, tmax, size=5, log=True):

    [ g, dg, gs, dgs, ddgs, gr ]=fcts
    lk = len(what_to_plot[0])
    lg = len(what_to_plot[1])

    fig = plt.figure(figsize=(size*1.33*lg, size*lk))

    for i in range(lk):
        
        key = what_to_plot[0][i]
        for l in range(len(what_to_plot[1])):

            (title, fc, yscale, ylim)= [ ("Total number of "+key, gs, ('log', {}), [10.0, 100.0, 10.0]),
                                         ("Speed (number of "+key+" by day) $\\left(\\frac{\\Delta "+key.replace(" ","\\_")+"}{\\Delta t}\\right)$", dgs, ('log', {}), [1.0, 10.0, 1.0]),
                                         ("Acceleration of the number of "+key+" $\\left(\\frac{\Delta^2 "+key.replace(" ","\\_")+"}{(\Delta t)^2}\\right)$", ddgs, ('symlog',{'linthreshy': 10.0}), [0,0,0]),
                                         ("Daily growth rate of "+key, gr, ('', {}), [0,0.5])
            ][ what_to_plot[1][l] ]

            ax = plt.subplot( lk, lg, lg*i + l+1 )
            plt.title(title, fontsize=int(size*2.7))
            
            for f in fc[key]:

                color = arg_plot[f]["color"]  # color and width of graph/text
                if f in focus:
                    lw=2
                else:
                    lw=1
                    
                if l==0:# and not log:    
                    
                    plt.plot( xr[key][f], g[key][f], "+", mew=lw/2.0, ms=lw*2.5, color=color )
                    
                if len(xr[key][f]) > l:
                    
                    plt.plot( xr[key][f][l:], fc[key][f][l:], lw=lw, **arg_plot[f] )
                    
                    if fc[key][f][-1] > ylim[i] or yscale[0]!="log" or not log:
                        if f in focus:
                            arg_text={"bbox": {'facecolor':'white', 'alpha':0.5, 'boxstyle':"round", 'edgecolor':color},
                                      "fontsize": 9,
                                      "color":color
                            }
                        else:
                            arg_text={"fontsize": 9,
                                      "color":color
                            }
                            
                        plt.text( xr[key][f][-1] , fc[key][f][-1], f, **arg_text )

            
            x1,x2 = xr[key][f][0],xr[key][f][-1]
            
            if yscale[0]=='': # gr
                for j in [2,3,4,5,6,7,10,20,30]:
                    z = pow(2.0,1.0/j)-1
                    plt.plot([x1, x2+(x2-x1)/5.0], [z]*2,"--",color='lightgrey', lw=1, zorder=0)
                    plt.text(x1,z,"$\\times 2$ in %d days"%j, fontsize=8, color="grey", zorder=0)
                plt.ylim(ylim[0],ylim[1])
                
            elif log:
                plt.yscale(yscale[0],**yscale[1])
                if key=='confirmed cases':
                    plt.ylim(bottom=100)
                else:
                    plt.ylim(bottom=10)

            plt.grid(True,which="both")
            plt.xlim(x1,x2+(x2-x1)/5.0) # room for country name
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return(fig)


#########################################
#### Main function


# Common part

def check_for_countries(threshold=death_threshold):

    data, days = get_data_from_files()
    data, xr = filter_by_var_threshold(data, days, "deaths", threshold, sync = False)
    
    print("\nWARNING: New countries to add to continents ?")
    for p in data["deaths"]:
        flag=False
        for r in regions:
            if p in regions[r]:
                flag=True
                break
        if not flag:
            print(p)

    print()
    
    
def prepare_data(begin="", end=""):
    
    data, days = get_data_from_files()  # get days
    if begin=="":
        begin = days[0]
    if end=="":
        end = days[-1]

    data, days = filter_by_dates(data, days, begin, end)  # get data filtered by filter by date
    add_regions(data)   # add all regions

    data, xr = filter_by_var_threshold(data, days, 'deaths', death_threshold, sync=False) # filter by number of deaths
    
    # sort the regions/coutries by number of deaths
    d = data["deaths"] 
    c = [x for x in d]
    mv = np.array([ max(d[x]) for x in d ])
    bests = list( mv.argsort()[:][::-1] )
    region_list = [ c[i] for i in bests ]

    # associate line colors and styles
    arg_plot= style_args(data, region_list)
    
    return data, days, xr, arg_plot

    

def prepare_graph(data, days, category, sync, log=False):

    if category=="top10":  # top10

        title = "'top-10' countries"
        data2 = filter_by_regions(data, ['World']+regions['World'], bool=False ) # remove regions
        d = data2["deaths"]
        c = [x for x in d]
        mv = np.array([ max(d[x]) for x in d ])
        bests = list( mv.argsort()[-10:][::-1] )
        
        region_list = [ c[i] for i in bests ]
        focus = []
        
    elif category in regions:    
                    
        title=category

        #        if not log:
        region_list = [i for i in regions[category]]+[category]
        focus=[category]
        #else:
        #    region_list = [i for i in regions[category]]
        #    focus=[]
            
    else:
        
        print("*** Wrong category:", category)
        exit(1)

    data2 = filter_by_regions(data, region_list)
        
    # copy (with potential synchronization)
    data2, xr2 = filter_by_var_threshold(data2, days, 'deaths', death_threshold, sync=sync) # filter by number of deaths
    
    #if sync:
    #    title+=", day 0: 1st day s.t. $deaths \\geq "+str(death_threshold)+"$"
    #title+=" (data from JHU CSSE on "+str(days[-1])+")"      

    tmax = max([ len(data2['deaths'][x]) for x in data2['deaths'] ] )
    
    return data2, xr2, region_list, focus, title, tmax


def get_filename(category,sync, log, what_to_plot):

    filename=category.replace(' ','_') + ["","_sync"][int(sync)] + ["","_log"][int(log)] + "_"
    for i in what_to_plot[0]:
        filename+=i[0]
    filename+="_"
    for i in what_to_plot[1]:
        filename+=str(i)
    return(filename)


def curves(category, what_to_plot=(keys, [0,1,2]), filename="", begin="", end="", sm=0, size=5, sync=False, log=True):

    filename=get_filename(category, sync, log, what_to_plot)
    
    print("Graph for "+category+", sync=",sync, "=>",filename)
        
    data, days, xr, arg_plot = prepare_data(begin=begin, end=end)
    
    data2, xr2, region_list, focus, title, tmax = prepare_graph(data, days, category=category, sync=sync, log=log)
    fcts = compute_fcts(data2, sm)
    fig = plot(region_list, what_to_plot, focus, xr2, fcts, arg_plot, tmax, size=size, log=log)
    plt.suptitle(title)#, fontsize = [7,10,10][len(what_to_plot[1])-1] )
    plt.savefig(filename+".png", dpi=dpi)
    plt.close('all')
    
    
def animation(category, what_to_plot=(keys, [0,1,2]), begin="", end="", sm=0, size=5, sync=False, log=True):
    
    filename=get_filename(category, sync, log, what_to_plot)+"_evol"
    
    print("Animation for "+category+" and several dates, sync=",sync,"=>",filename)

    data, days, xr, arg_plot = prepare_data(begin=begin, end=end)
    data2, xr2, region_list, focus, title, tmax = prepare_graph(data, days, category=category, sync=sync, log=log)
    fcts = compute_fcts(data, sm)
        
    for t in range(1,tmax):
        
        print("t=",t,"day=",days[t])

        # Crop all data 
        fcts2 = [ dict() for j in range(len(fcts)) ]
        xr2 = dict()
        for j in range(len(fcts)):
            for key in keys:
                fcts2[j][key] = dict()
                if j==0:
                    xr2[key] = dict()
                for f in data2[key]:
                    z = max(0, len(data2[key][f]) - tmax + t + 1)
                    fcts2[j][key][f] = fcts[j][key][f][0:z]
                    if j==0:
                        xr2[key][f] = xr[key][f][0:z]
                    
        fig = plot(region_list, what_to_plot, focus, xr2, fcts2, arg_plot, tmax, size=size, log=log)
        plt.suptitle(title)
        plt.savefig("fig/"+filename+"_%02d.png"%t, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 10 -loop 0 "
    ltmax = list( range( 3,tmax-1 ) ) 
    src = " ".join([ "fig/"+filename+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    #os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".mp4" )
    os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".gif" )


#########################
# Main global parameters



check_for_countries()

dpi = 90 # graph quality (for png/mp4)

#what_to_plot=[ keys, [0,1,2] ]
for what_to_plot in [ [ ['deaths' ], [0,3] ],
                      [ ['confirmed cases' ], [0,3] ]#,
#                      [ ['recovered cases' ], [0,3] ]
]:

    for (region, begin) in [
            ('World',''),
            ('top10',''),
            ('Europe','2/18/20'),
            ('Asia',''),
            ('North America','2/27/20'),
            ('South America','3/15/20'),
            ('Africa','3/9/20')#,
            #('Australia','')        
            ]:
        curves(region, what_to_plot=what_to_plot, begin=begin, sm=5)               
        #    curves(region, what_to_plot=what_to_plot, sync=True)                  
        #animation(region, what_to_plot=what_to_plot, begin=begin, sm=15) 

