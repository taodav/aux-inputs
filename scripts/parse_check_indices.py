
if __name__ == "__main__":
    job_to_run = ""
    indices = [
        1,2,3,4,5,8,9,10,14,15,16,17,18,19,20,21,22,23,24,25,26,27,32,33,34,35,36,37,38,39,42,43,44,48,49,50,51,52,54,56
    ]

    sorted_indices = list(sorted(indices))

    dashed_indices = []
    curr_seq_indices = [sorted_indices[0]]

    for i, idx in enumerate(sorted_indices):
        if i > 0:
            if idx - 1 == curr_seq_indices[-1]:
                curr_seq_indices.append(idx)
            else:
                # in this case, we have a break in sequential indices
                if len(curr_seq_indices) > 1:
                    dashed_indices.append(f"{curr_seq_indices[0]}-{curr_seq_indices[-1]}")
                else:
                    # in this case, there was only a single element in dashed_indices
                    dashed_indices.append(str(curr_seq_indices[0]))
                curr_seq_indices = [idx]
    else:
        dashed_indices.append(f"{curr_seq_indices[0]}-{curr_seq_indices[-1]}")

    print(",".join(dashed_indices))
    print(f"In all, there are {len(sorted_indices)} job(s) to run.")
