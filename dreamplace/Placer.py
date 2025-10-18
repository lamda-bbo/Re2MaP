##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import json
import logging
import shutil
import math
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.abspath('.'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(parent_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
import torch

import Cluster
import networkit as nk
from Plot import extract_adjacency_matrix, plot_ports, plot_clusters

from remap_.ClusterPlacer import CoordinateClusterPlacer, ABPlacer
from remap_.MacroDistributor import NaiveDistributor, GridGuideDistributor
from remap_.utils.def2mp import def2mp
from datetime import datetime

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_gp_result(placer, path, figname="gp"):
    figname = "%s/%s.png" % (path, figname)
    os.system("mkdir -p %s" % (os.path.dirname(figname)))
    pos = placer.pos[0].data.clone().cpu().numpy()
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    placer.op_collections.draw_place_op(pos, figname)


def place(params):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    set_seed(params.random_seed)
    
    runtime_record = [{"I/O": 0, "DMP-Gradient": 0, "ABPlace": 0, "MacroDistribution": 0, "Extraction": 0}]
    tot_time = time.time()
    def update_runtime(event, start_time):
        runtime_record[-1][event] += time.time() - start_time
    
    def shift_runtime_record():
        runtime_record.append(runtime_record[-1].copy())
    
    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    update_runtime("I/O", tt)
    logging.info("reading database takes %.2f seconds" % (time.time() - tt))
    


    # testing clustering algorithm
    cell_cluster = Cluster.Cluster(placedb, "benchmarks/clustering_results/ariane133/cluster_map.txt", "benchmarks/clustering_results/ariane133/clustering_results.json", "benchmarks/clustering_results/ariane133/dataflow_results.json")

    pdb.set_trace()

    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

        # Dump example here. Some dump functions are defined.
        # Check instance methods defined in Timer.py for debugging.
        # timer.dump_pin_cap("pin_caps.txt")
        # timer.dump_graph("timing_graph.txt")

    if params.remap_flag:
        remap_flag = True
        # set remap_flag to 0 in order to run pure DREAMPlace non-linear placement flow for initialization
        params.remap_flag = 0

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    metrics = placer(params, placedb)
    
    update_runtime("DMP-Gradient", tt)
    logging.info("non-linear placement takes %.2f seconds" %
                 (time.time() - tt))

    if remap_flag:
        macro_halo_x = params.macro_halo_x
        macro_halo_y = params.macro_halo_y
        design_name = params.design_name()
        now = datetime.now()
        tymd = now.strftime("%Y%m%d")
        thms = now.strftime("%H%M%S")
        if params.save2result_dir:
            path = params.result_dir
        else:
            path = os.path.join(
                params.result_dir, tymd, design_name, thms)
        if not os.path.exists(path):
            os.system("mkdir -p %s" % (path))
        # recover remap_flag, from now on, DREAMPlace non-linear placement flow ends early
        params.remap_flag = 1
        # initialization
        layout_info = {
            "xl": placedb.xl,
            "xh": placedb.xh,
            "yl": placedb.yl,
            "yh": placedb.yh
        }
        macro_id = np.where(placedb.movable_macro_mask)[0]
        num_macros = len(macro_id)
        macro_w = placedb.node_size_x[macro_id]
        macro_h = placedb.node_size_y[macro_id]
        movable_macro_mask = np.zeros(placedb.num_nodes, dtype=bool)
        movable_macro_mask[:num_macros] = 1
        
        if params.macros2place_each_step == 0:
            params.macros2place_each_step = \
                math.ceil(num_macros / 10)
        
        expected_steps = math.ceil(num_macros / params.macros2place_each_step)
        
        if params.target_density_lb < 0:
            target_density_decay = 1.0
        else:
            target_density_decay = \
                (params.target_density_lb / params.target_density) \
                    ** (1 / expected_steps)
        
        radius_ratio = params.init_radius_ratio
        radius_ratio_gamma = (params.radius_ratio_lb / radius_ratio) ** (1 / expected_steps)
        
        # pdb.set_trace()
        tt = time.time()
        builder = Cluster.GraphBuilder(placedb)
        g = builder.build_graph()
        partition_path = "%s/%s/%s" % (params.result_dir, design_name, ".partition")
        if not os.path.exists(partition_path):
            plm = nk.community.PLM(g, gamma=10, par='none')
            plm.run()
            communities = plm.getPartition()
            nk.community.writeCommunities(communities, partition_path)
        else:
            communities = nk.community.readCommunities(partition_path)
        print("Number of communities:", communities.numberOfSubsets())

        adjacency_matrix, clusters, node2cluster, \
            cluster_x, cluster_y, second_order_matrix, third_order_matrix = \
                extract_adjacency_matrix(placedb, communities, g, macro_id)
        
        # adjacency_matrix = adjacency_matrix.astype(dtype=np.float64) + second_order_matrix
        # adjacency_matrix = adjacency_matrix / (np.max(adjacency_matrix) + 1e-5)
        # assert 0, np.max(adjacency_matrix)
        
        cluster2node_names = []
        for nodes in clusters:
            cluster2node_names.append(
                [placedb.node_names[node].decode(encoding='utf-8')
                 for node in nodes if node < len(placedb.node_x)])

        assert 0, (len(placedb.node_x), placedb.num_nodes)

        update_runtime("Extraction", tt)

        if params.remap_plot_flag:
            plot_gp_result(placer, path)

        it = 0
        num_nodes = len(clusters)
        cluster_x = np.array(cluster_x)
        cluster_y = np.array(cluster_y)
        
        tt = time.time()
        
        abpl = ABPlacer(
            adjacency_matrix, macro_w, macro_h, layout_info,
            device="cuda:0",
        )
        abpl(cluster_x, cluster_y, movable_macro_mask[:num_nodes])
        theta = \
            abpl.place(halo=0, l_overlap_penalty=params.l_overlap_penalty)

        update_runtime("ABPlace", tt)
        
        os.system(f"mkdir -p {path}/{it}")
        
        if params.remap_plot_flag:
            abpl.plot_distribution(theta, 
                figname=f"{path}/{it}/0_post_mp_0_distribution")
            
        movable_macro_x, movable_macro_y = \
            abpl.theta2coordinate(theta, ratio=radius_ratio)
        radius_ratio *= radius_ratio_gamma
        
        cluster_x[movable_macro_mask[:num_nodes]] = movable_macro_x
        cluster_y[movable_macro_mask[:num_nodes]] = movable_macro_y
        
        if params.remap_plot_flag:
            abpl.plot_layout(xy=(cluster_x, cluster_y), 
                figname=f"{path}/{it}/0_post_mp_1_layout")
            abpl.plot_layout(xy=(cluster_x, cluster_y),
                    figname=f"{path}/layout")

        tt = time.time()

        ggdb = GridGuideDistributor(adjacency_matrix, macro_w, macro_h, layout_info, macro_halo_x=macro_halo_x, macro_halo_y=macro_halo_y)
        ggdb(cluster_x, cluster_y, movable_macro_mask[:num_macros])
        cluster_x, cluster_y, unplaced_macro_mask = \
            ggdb.distribute(params.macros2place_each_step)
        movable_macro_mask[:num_macros] = unplaced_macro_mask
        
        update_runtime("MacroDistribution", tt)

        if params.remap_plot_flag:
            abpl(cluster_x, cluster_y, movable_macro_mask[:num_nodes])
            abpl.plot_layout(xy=(cluster_x, cluster_y),
                figname=f"{path}/{it}/1_post_distribution")
            abpl.plot_layout(xy=(cluster_x, cluster_y),
                figname=f"{path}/layout")

        # params.random_center_init_flag = 0
        
        tt = time.time()
        
        params.plot_flag = 0
        params.macro_halo_x = 0
        params.macro_halo_y = 0
        placedb.node_x[macro_id] = cluster_x[:num_macros] - placedb.node_size_x[macro_id] / 2
        placedb.node_y[macro_id] = cluster_y[:num_macros] - placedb.node_size_y[macro_id] / 2
        placedb.apply(params, placedb.node_x, placedb.node_y)
        placedb.movable_macro_mask[macro_id] = ~movable_macro_mask[:num_macros]
        
        gp_out_file = os.path.join(
            path,
            "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
        placedb.write(params, gp_out_file, blockages=ggdb.generate_blockages(params))
        params.def_input = gp_out_file
        
        update_runtime("I/O", tt)
        shift_runtime_record()
        
        while np.any(movable_macro_mask[:num_macros]):
            it += 1
            tt = time.time()
            params.target_density *= target_density_decay
            placedb = PlaceDB.PlaceDB()
            placedb(params)
            update_runtime("I/O", tt)
            del placer
            tt = time.time()
            placer = NonLinearPlace.NonLinearPlace(params, placedb, timer=None)
            metrics = placer(params, placedb)   
            update_runtime("DMP-Gradient", tt)
            
            if params.remap_plot_flag:
                plot_gp_result(placer, path)


            tt = time.time()
            macro_id = []
            cluster_x = []
            cluster_y = []
            for cluster_id, node_names in enumerate(cluster2node_names):
                node_x = []
                node_y = []
                for node_name in node_names:
                    possible_keys = [
                        node_name,
                        f"{node_name}.DREAMPlace.Shape0"
                    ]
                    for possible_key in possible_keys:
                        if possible_key in placedb.node_name2id_map:
                            node_id = placedb.node_name2id_map[possible_key]
                            break
                    else:
                        node_id = -1
                        print([(bn).decode('utf-8')
                               for bn in placedb.node_names
                               if len(bn.decode()) >= len(node_name)
                               and bn.decode().startswith(node_name)])
                        print(node_name)
                        assert 0
                    if 0 <= node_id < len(placedb.node_x):
                        node_x.append(placedb.node_x[node_id] 
                                    + placedb.node_size_x[node_id] / 2)
                        node_y.append(placedb.node_y[node_id] 
                                    + placedb.node_size_y[node_id] / 2)
                        if cluster_id < num_macros:
                            assert len(node_names) == 1
                            macro_id.append(node_id)
                avg = lambda lst: sum(lst) / len(lst)
                cluster_x.append(avg(node_x))
                cluster_y.append(avg(node_y))
                
            update_runtime("Extraction", tt)

            macro_id = np.array(macro_id)
            assert len(macro_id) == num_macros
            cluster_x = np.array(cluster_x)
            cluster_y = np.array(cluster_y)
            
            tt = time.time()
            
            abpl(cluster_x, cluster_y, movable_macro_mask[:num_nodes])
            theta = abpl.place(halo=0, l_overlap_penalty=params.l_overlap_penalty)
            
            update_runtime("ABPlace", tt)
            
            os.system(f"mkdir -p {path}/{it}")
            
            if params.remap_plot_flag:
                abpl.plot_distribution(theta, 
                    figname=f"{path}/{it}/0_post_mp_0_distribution")
                
            movable_macro_x, movable_macro_y = \
                abpl.theta2coordinate(theta, ratio=radius_ratio)
            radius_ratio *= radius_ratio_gamma
            
            cluster_x[movable_macro_mask[:num_nodes]] = movable_macro_x
            cluster_y[movable_macro_mask[:num_nodes]] = movable_macro_y
            
            if params.remap_plot_flag:
                abpl.plot_layout(xy=(cluster_x, cluster_y), 
                    figname=f"{path}/{it}/0_post_mp_1_layout")
                abpl.plot_layout(xy=(cluster_x, cluster_y),
                    figname=f"{path}/layout")

            tt = time.time()

            ggdb(cluster_x, cluster_y, movable_macro_mask[:num_macros])
            cluster_x, cluster_y, unplaced_macro_mask = \
                ggdb.distribute(params.macros2place_each_step)
                
            update_runtime("MacroDistribution", tt)
            movable_macro_mask[:num_macros] = unplaced_macro_mask

            if params.remap_plot_flag:
                abpl(cluster_x, cluster_y, movable_macro_mask[:num_nodes])
                abpl.plot_layout(xy=(cluster_x, cluster_y),
                    figname=f"{path}/{it}/1_post_distribution")
                abpl.plot_layout(xy=(cluster_x, cluster_y),
                    figname=f"{path}/layout")

            tt = time.time()
            
            placedb.node_x[macro_id] = cluster_x[:num_macros] - macro_w / 2
            placedb.node_y[macro_id] = cluster_y[:num_macros] - macro_h / 2
            placedb.apply(params, placedb.node_x, placedb.node_y)
            movable_macros = macro_id < placedb.movable_macro_mask.shape[0]
            movable_macro_id = macro_id[movable_macros]
            placedb.movable_macro_mask[movable_macro_id] = \
                ~movable_macro_mask[:num_macros][movable_macros]
            placedb.write(params, gp_out_file, blockages=ggdb.generate_blockages(params))
            
            update_runtime("I/O", tt)
            shift_runtime_record()
        
        runtime_record[-1]["Total"] = time.time() - tot_time
        
        with open(f"{path}/runtime.json", "w") as f:
            f.write(json.dumps(
                runtime_record,
                sort_keys=False,
                indent=4,
                separators=(",", ": ")
            ))
        
        ref_path = f"{path}/ref_mp_out" \
                   if params.ref_path == "" \
                   else params.ref_path
        def2mp(gp_out_file, ref_path, f"{path}/mp_out")
        
        ## END ReMaP flow
    else:
        path = "%s/%s" % (params.result_dir, params.design_name())
        if not os.path.exists(path):
            os.system("mkdir -p %s" % (path))
        gp_out_file = os.path.join(
            path,
            "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
        placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(
            params.detailed_place_engine):
        logging.info("Use external detailed placement engine %s" %
                     (params.detailed_place_engine))
        if params.solution_file_suffix() == "pl" and any(
                dp_engine in params.detailed_place_engine
                for dp_engine in ['ntuplace3', 'ntuplace4h']):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine, params.aux_input, gp_out_file,
                target_density_cmd, dp_out_file, params.detailed_place_command)
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0:placedb.num_physical_nodes] = placedb.node_x
                pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_physical_nodes] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration)
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density))
                placer.plot(params, placedb, iteration, pos)
        elif 'ntuplace_4dr' in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            if(params.verilog_input):
                cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(params.verilog_input))
                benchmark_dir)
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found" %
            (params.detailed_place_engine))

    return metrics


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()

    # load parameters
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    place(params)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))
