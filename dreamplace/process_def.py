def remove_blockage(file_path):
    start_phrase = "BLOCKAGES"
    end_phrase = "END BLOCKAGES"
    delete_blockage = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if start_phrase in line:
                delete_blockage = True
                print("find_blockage")
            if not delete_blockage:
                file.write(line)
            if end_phrase in line:
                delete_blockage = False
                print("over_blockage")

def from_fixed_to_placed(file_path):
    modified_lines = []
    replace_count = 0

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if "fakeram" in line and "FIXED" in line:
                original_line = line
                line = line.replace("FIXED", "PLACED")
                if line != original_line:
                    replace_count += 1
            modified_lines.append(line)
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
        print(f"File '{file_path}' has been successfully modified. Total replacements: {replace_count}.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def fix_certain_macros(file_path, macro_names):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    output_lines = []
    replace_next = False
    replace_count = 0
    
    for i, line in enumerate(lines):
        if replace_next:
            if "PLACED" in line:
                line = line.replace("PLACED", "FIXED")
                replace_count += 1
            replace_next = False
        for macro_name in macro_names:
            name = macro_name.decode(encoding='utf-8')
            if name in line:
                replace_next = True
        output_lines.append(line)
    print("replace ", replace_count, " fixed")

    with open(file_path, 'w') as file:
        file.writelines(output_lines)

def modify_blockages(file_path, blockages, replace=True):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    blockage_start = None
    blockage_head = None
    blockage_end = None
    pin_end = None
    
    for index, content in enumerate(lines):
        if content.startswith("BLOCKAGES") and blockage_start is None:
            blockage_start = index
            blockage_head = content
            num_blockages = int(content.rstrip().rstrip(";").split()[1])
        if content.startswith("END BLOCKAGES") and blockage_end is None:
            blockage_end = index
        if content.startswith("END PINS") and pin_end is None:
            pin_end = index
    
    def extract(blockages):
        segment = []
        for blockage in blockages:
            xl, yl, xh, yh = blockage
            segment.append(
                f"    - PLACEMENT + SOFT RECT ( {xl} {yl} ) ( {xh} {yh} ) ;\n"
            )
        return segment
    
    END_BLOCKAGES = "END BLOCKAGES\n"
    
    new_lines = []
    if blockage_head is None:
        new_lines.extend(lines[:pin_end+1])
        blockage_segment = [f"BLOCKAGES {len(blockages)} ;\n"]
        blockage_segment.extend(extract(blockages))
        blockage_segment.append(END_BLOCKAGES)
        new_lines.extend(blockage_segment)
        if pin_end < len(lines):
            new_lines.extend(lines[pin_end+1:])
    else:
        if replace:
            new_lines.extend(lines[:blockage_start])
            blockage_segment = [f"BLOCKAGES {len(blockages)} ;\n"]
            blockage_segment.extend(extract(blockages))
            blockage_segment.append(END_BLOCKAGES)
            new_lines.extend(blockage_segment)
            if blockage_end < len(lines):
                new_lines.extend(lines[blockage_end+1:])
        else:
            new_lines.extend(lines[:blockage_start])
            blockage_segment = \
                [f"BLOCKAGES {len(blockages) + num_blockages} ;\n"]
            if blockage_start + 1 < blockage_end:
                blockage_segment.extend(
                    lines[blockage_start+1:blockage_end])
            blockage_segment.extend(extract(blockages))
            blockage_segment.append(END_BLOCKAGES)
            new_lines.extend(blockage_segment)
            if blockage_end < len(lines):
                new_lines.extend(lines[blockage_end+1:])
    
    with open(file_path, "w") as f:
        f.writelines(new_lines)
            

# if __name__ == '__main__':
#     designs = ["ariane133", "ariane136", "black_parrot", "bp_be", "bp_fe", "bp_multi", "bp_quad", "swerv_wrapper"]
#     for design in designs:
#         remove_blockage(f'install/benchmarks/or_cases/{design}/2_5_floorplan_macro.def')
#         from_fixed_to_placed(f'install/benchmarks/or_cases/{design}/2_5_floorplan_macro.def')
