import os
import argparse

import common

import logging

from dreamplace.PlaceDB import PlaceDB
from dreamplace.Params import Params

from remap import DataWrapper, ReMaP, GlobalInfo
from remap.clustering import LouvainClusterer, MixedClusterer
from remap.common.timer import Timer

timer = Timer()

@timer.listen("ReMaP flow")
def remap_flow(args):
    g = GlobalInfo()
    g.seed = args.seed
    
    params = Params()
    params.load(args.json_config)
    params.macro_place_flag = 0
    
    g.design_name = os.path.splitext(os.path.basename(args.json_config))[0]
    if args.tag == "":
        g.run_tag = g.design_name
    else:
        g.run_tag = args.tag
        
    if args.workspace == "":
        g.workspace = params.result_dir
    else:
        g.workspace = args.workspace
    
    i_o_timer = timer.listen_handler("I/O")
    i_o_hdlr = next(i_o_timer)
    placedb = PlaceDB()
    placedb(params)
    i_o_hdlr()
    
    # clusterer = LouvainClusterer(placedb)
    clustering_timer = timer.listen_handler("Clustering")
    clustering_hdlr = next(clustering_timer)
    clusterer = MixedClusterer(placedb, params)
    clustering_hdlr()
    
    datawrapper = DataWrapper(placedb, params, clusterer)
    datawrapper.from_placedb()
    
    placer = ReMaP()
    placer.bind(datawrapper)
    placer.place(
        radius_ratio_init=params.radius_ratio_init,
        radius_ratio_lb=params.radius_ratio_lb,
        target_density_init=params.target_density_init,
        target_density_lb=params.target_density_lb
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, metavar='<config>')
    parser.add_argument('-t', '--tag', type=str, dest='tag', metavar='<tag>', default="")
    parser.add_argument('-w', '--workspace', type=str, dest='workspace', metavar='<path>', default="")
    parser.add_argument('-s', '--seed', type=int, dest='seed', metavar='<seed>', default=1)
    args = parser.parse_args()
    
    remap_flow(args)
    timer.report(logging_fn=logging.info)