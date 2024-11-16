import xml.etree.ElementTree as ET
import os
import re
from collections import defaultdict
import pandas as pd


def get_blocks_list(block_infos_file_path: str) -> list:
    """
    build the blocks dictionary list from block.infos file. The length of list is the number of blocks in the netlist, and the each element is one dictionary recorded block infomation.
    The dictionary include 3 keys, including name, type, index.
    Input: the blocks.infos file name
    Output: the blocks information list
    """
    blocks_list = []
    with open(block_infos_file_path) as file:
        for index, line in enumerate(file.readlines()):
            if index == 0:
                continue

            block_dict = {}
            block_dict["name"] = line.strip().split(" ")[0]
            block_dict["type"] = line.strip().split(" ")[-1]
            block_dict["index"] = index - 1
            block_dict["width"] = 0
            block_dict["height"] = 0
            block_dict["connections"] = 0
            block_dict["source"] = 0
            block_dict["sink"] = 0
            blocks_list.append(block_dict)
    return blocks_list


def write_blank_place_file(blocks: list, file_path: str):
    """
    generate the blank place file.
    Input:
        blocks: blocks list returned from get_blocks_list function
        file_path: the output filename named with .place suffix
    Output:
        generate the blank place file
    TODO: add the titile of the placement information( need to retrieve from VTR source code)
    """
    with open(file_path, "w") as file:
        file.write(
            "Netlist_File: tseng.net Netlist_ID: SHA256:a3c4b29e34465eee0ccf98fd1cfbf7a2be835aaf1683064cb2ac81ac5f869669\n"
        )
        file.write("Array size: 11 x 11 logic blocks\n\n")
        file.write("#block name\tx\ty\tsubblk\tblock number\n")
        file.write("#----------\t--\t--\t------\t------------\n")
        for index, block in enumerate(blocks):
            # if index <= 55:
            file.write("{}\t\t0\t0\t0\t#{}\n".format(block["name"], block["index"]))


def get_grid_infos(grid_constraint_file_path) -> (dict, int, int, int):  # type: ignore
    """
    constraint dictionary indicated the possible positions of the blocks placement
    the key is the type of blocks, and the corresponding value indicates the possible placement positions
    Input: the grid.constraint inforamtion returned from VPR pack process
    Output: grid_constraint_dict
    """
    grid_constraint_dict = {}
    block_size = {}
    capacity = 1

    with open(grid_constraint_file_path, "r") as file:
        for index, line in enumerate(file.readlines()):
            if index == 0:
                grid_width = int(line.strip().split(" ")[1])
                grid_height = int(line.strip().split(" ")[3])

            if index <= 1:
                continue

            line = re.sub(" +", " ", line).strip()
            line_split = line.split(" ")
            capacity = (
                int(line_split[-2]) if capacity < int(line_split[-2]) else capacity
            )

            if line_split[0] not in grid_constraint_dict:

                grid_constraint_dict[line_split[0]] = [
                    int(line_split[1]) * grid_width + int(line_split[2])
                ]

                block_size[line_split[0]] = [line_split[3], line_split[4]]

            else:
                grid_constraint_dict[line_split[0]].append(
                    int(line_split[1]) * grid_width + int(line_split[2])
                )

    return grid_constraint_dict, grid_width, grid_height, capacity, block_size


def __find_atoms(subblocks, root):
    for child in root.findall("block"):
        subblocks.append(child.get("name"))
        __find_atoms(subblocks, child)


def __find_block(root) -> dict:
    blocks = {}
    for child in root.findall("block"):
        subblocks = []
        __find_atoms(subblocks, child)
        blocks[child.get("name")] = list(dict.fromkeys(subblocks))
    return blocks


def get_block_primitives_dict(xml_path) -> dict:
    """
    build the blocks packing dictionary recorded which primitives are packed into the block.
    the key is the block name and the corresponding value is the primitives packed inside.
    Input: vpr pack process output(the file with .net suffix).
    Output: block_primitives dictionary
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    blocks_primitives_dict = __find_block(root)
    return blocks_primitives_dict


def get_netlist_dict(
    blocks_primitives_dict: dict,
    blocks_list: list,
    primitive_netlist_file_path="primitive.netlist",
) -> dict:
    """
    generate netlists dictionary. the key is the net name, the value is the list of blocks index in the net.
    Input:
        blocks_primitives_dict, returned from function get_blocks_primitives_dict
        blocks_list, returned from function get_block_list
        primitive_netlist_file_path, the file path of the primitives netlist
    Output:
        netlist_dict
    """
    netlist_dict = {}

    with open(primitive_netlist_file_path, "r") as file:
        for index, line in enumerate(file.readlines()):
            # skip the first line
            if index == 0:
                continue

            net = line.strip().split(" ")[-1]
            primitive = line.strip().split(" ")[0]

            # retrieve the index of the block containing the specificed primitive
            for key, value in blocks_primitives_dict.items():
                if primitive in value:
                    for block in blocks_list:
                        if block["name"] == key:
                            index = block["index"]

            # # allow the duplicated block show more than 1 time, i.e. [1, 3, 4, 3, 1]
            # # instead of removing all duplicate blocks in one netlist i.e. [1, 3, 4]
            # if net in netlist_dict.keys():
            #     # if the consecutive primitives are inside the same block, only keep one
            #     # i.e. [1, 3, 4, 3, 3, 3, 1] --> [1, 3, 4, 3, 1]
            #     if netlist_dict[net][-1] == index:
            #         continue
            #     else:
            #         netlist_dict[net].append(index)
            # else:
            #     netlist_dict[net] = [index]

            # remove the sinks same with source, i.e. [1, 2, 1, 1] --> [1, 2]; [7, 7, 7] --> [7]
            if net in netlist_dict.keys():
                if index == netlist_dict[net][0] or index in netlist_dict[net]:
                    continue
                else:
                    netlist_dict[net].append(index)
            else:
                netlist_dict[net] = [index]

    return netlist_dict


def get_netlist_list(netlist_dict: dict) -> list:
    """
    return the list of netlist, which removes all singleton
    Input: netlist_dict returnd from the function get_netlist_dict
    Output: netlist_list
    """
    netlist_list = []

    # remove all singleton and return a lsit
    for key, value in netlist_dict.items():
        if len(value) == 1:
            continue
        else:
            netlist_list.append(value)

    return netlist_list


def get_netlist_list_removing_duplicates(netlist_list: list) -> list:
    """
    remove all duplicated netlist in the list. We define duplicates as:
        1. if the net length is 2, the order of the source and sink is not matter. e.g. [1, 2] and [2, 1] are duplicates.
        2. the nets are exactly same. e.g. [1, 2, 3, 4] and [1, 2, 3, 4] are duplicates.
    """
    edge_set = set()
    for net in netlist_list:
        if len(net) == 2:
            edge_set.add(tuple(sorted(net)))
        else:
            edge_set.add(tuple(net))

    return list(edge_set)


def set_place_order(blocks_list, num_placed_blocks, type="default"):
    if type == "default":
        return blocks_list[:num_placed_blocks]["index"].to_list()
    elif type == "connections":
        sorted_df = blocks_list[:num_placed_blocks].sort_values(
            by="connections", ascending=False
        )
        return sorted_df["index"].to_list()


class Preprocess:
    def __init__(
        self,
        num_target_blocks,
        pack_xml_path,
        block_infos_file_path,
        primitive_netlist_file_path,
        grid_constraint_path,
        blocks_place_file_path,
        order="connections",
    ) -> None:
        self.grid_constraint_path = grid_constraint_path

        """
        TODO netlist_file and netlist_id are blank from pack.cpp
        """
        self.netlist_file = None
        self.netlist_id = None
        self.capacity = 0
        self.num_target_blocks = num_target_blocks

        self.blocks_list = get_blocks_list(block_infos_file_path)
        write_blank_place_file(self.blocks_list, blocks_place_file_path)
        (
            self.grid_constraints_dict,
            self.grid_width,
            self.grid_height,
            self.capacity,
            block_size,
        ) = get_grid_infos(grid_constraint_path)
        self.blocks_primitives_dict = get_block_primitives_dict(pack_xml_path)
        self.netlist_dict = get_netlist_dict(
            self.blocks_primitives_dict,
            self.blocks_list,
            primitive_netlist_file_path,
        )
        self.netlist_list = get_netlist_list(self.netlist_dict)
        for block_infos in self.blocks_list:
            block_infos["width"] = int(block_size[block_infos["type"]][0])
            block_infos["height"] = int(block_size[block_infos["type"]][1])
            for list in self.netlist_list:
                if list[0] == block_infos["index"]:
                    block_infos["connections"] += len(list) - 1
                elif block_infos["index"] in list and list[0] != block_infos["index"]:
                    block_infos["connections"] += 1

            block_infos["source"] = sum(
                1 for sublist in self.netlist_list if sublist[0] == block_infos["index"]
            )

            block_infos["sink"] = sum(
                1
                for sublist in self.netlist_list
                if block_infos["index"] in sublist
                and sublist[0] != block_infos["index"]
            )

        # if order == "connection":
        #     self.blocks_list = sorted(
        #         self.blocks_list, key=lambda x: x["connections"], reverse=True
        #     )
        # else:
        #     pass
        # for i, item in enumerate(self.blocks_list):
        #     item["order"] = i

        self.blocks_list = pd.DataFrame(self.blocks_list)
        self.place_order = set_place_order(
            self.blocks_list, num_target_blocks, type=order
        )

        # self.netlist_list_removing_duplicates = (
        #     preprocess.get_netlist_list_removing_duplicates(netlist_list)
        # )

    # def get_netlist_list_removing_duplicates(self, netlist_list: list) -> list:
    #     """
    #     remove all duplicated netlist in the list. We define duplicates as:
    #         1. the netlist with the same source but with the subset of the sinks(the order of the source is not matter).
    #             E.g. [1, 2, 3, 4] and [1, 4, 3] are duplicates.
    #         2. the netlist with the same source and sinks but with different order.
    #             E.g. [1, 2, 3, 4] and [1, 4, 3, 2] are duplicates.
    #     Input: netlist list returned from the function get_netlist_list
    #     Output: netlist list removing all duplicates
    #     """
    #     # Create a dictionary to store the elements and their corresponding sublists
    #     elements_dict = defaultdict(list)
    #     elements_dict_ = defaultdict(list)

    #     # Iterate through each sublist in the original list
    #     for sublist in netlist_list:
    #         for element in sublist:
    #             elements_dict[element].append(sublist)

    #         elements_dict_[sublist[0]].append(sublist)

    #     # Create a set to store the unique sublists
    #     unique_sublists = set()

    #     # Add the longest sublists to the set
    #     for sublists, sublists_ in zip(elements_dict.values(), elements_dict_.values()):

    #         max_length = max(len(sublist) for sublist in sublists)
    #         # Find all sublists with the maximum length
    #         longest_sublists = [sublist for sublist in sublists if len(sublist) == max_length]
    #         for longest_sublist in longest_sublists: unique_sublists.add(tuple(longest_sublist))

    #         max_length = max(len(sublist) for sublist in sublists_)
    #         # Find all sublists with the maximum length
    #         longest_sublists_ = [sublist for sublist in sublists_ if len(sublist) == max_length]
    #         for longest_sublist_ in longest_sublists_: unique_sublists.add(tuple(longest_sublist_))

    #     # Convert the set of tuples back to a list of lists
    #     new_list = [list(sublist) for sublist in unique_sublists]

    #     return new_list


if __name__ == "__main__":
    preprocess = Preprocess(
        5,
        os.path.join("/home/swang848/RL-FPGA/", "data/tseng.net"),
        os.path.join("/home/swang848/RL-FPGA/", "data/block.infos"),
        os.path.join("/home/swang848/RL-FPGA/", "data/primitive.netlist"),
        os.path.join("/home/swang848/RL-FPGA/", "data/grid.constraint"),
        os.path.join("/home/swang848/RL-FPGA/", "data/tseng.place"),
    )
    # grid_constraints_dict = process.get_grid_constraints_dict()
    # print(grid_constraints_dict)
    # print(process.capacity)
    block_list = preprocess.blocks_list
    print(preprocess.grid_constraints_dict)
    print(block_list.loc[block_list["index"] == 10])
    # print(block_list)
    # print(preprocess.grid_constraints_dict)
    # netlist_dict = preprocess.netlist_dict

    # netlist_list_removing_duplicates = preprocess.get_netlist_list_removing_duplicates(
    #     netlist_list
    # )
    # print(netlist_list_removing_duplicates)
    # print(len(netlist_list_removing_duplicates))

    # a = set()
    # for i in netlist_list_removing_duplicates:
    #     for j in i:
    #         a.add(j)
    # print(a)

    # edge_set = set()
    # edge_dict = dict()

    # # add the edge into the edge_dict and count the number of connections between nodes
    # for value_list in netlist_list:
    #     for i in range(len(value_list) - 1):
    #         edge_set.add((value_list[0], value_list[i + 1]))
    #         if (value_list[0], value_list[i + 1]) not in edge_dict.keys():
    #             edge_dict[(value_list[0], value_list[i + 1])] = 1
    #         else:
    #             edge_dict[(value_list[0], value_list[i + 1])] += 1

    # # make the edges undirected
    # for key in edge_dict.keys():
    #     if (key[1], key[0]) in edge_dict.keys() and key[1] != key[0]:
    #         edge_dict[key] += edge_dict[(key[1], key[0])]
    #         edge_dict[(key[1], key[0])] = 0
    #     else:
    #         pass

    # # remove all keys with value 0
    # edge_dict = {x: y for x, y in edge_dict.items() if y != 0}
    # import torch
    # from torch_geometric.data import Data

    # # build pytorch geometric undirected graph
    # edge_index = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
    # edge_index_reverse = edge_index.clone()
    # edge_index_reverse[[0, 1]] = edge_index_reverse[[1, 0]]
    # edge_weight = torch.tensor(list(edge_dict.values()), dtype=torch.float)

    # edge_index = torch.cat((edge_index, edge_index_reverse), 1)
    # edge_weight = torch.cat((edge_weight, edge_weight), 0)
    # x = torch.full((230, 22), -1, dtype=torch.float)
    # graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # graph_data.validate()
