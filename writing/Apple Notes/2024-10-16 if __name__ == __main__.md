if __name__ == "__main__":        
    if len(sys.argv)==6:
        filename = sys.argv\[1\]
        startframe = int(sys.argv\[2\])
        endframe = int(sys.argv\[3\])
        suffix = sys.argv\[4\]
        dest_dir = sys.argv\[5\]
    else:
        print("usage: python cut_bvh.py startframe endframe suffix dest_dir")
        sys.exit(0)
        
    #import pdb;pdb.set_trace()
    print(f'Cutting BVH {filename} from: {startframe} to {endframe}')
    basename = os.path.splitext(os.path.basename(filename))\[0\]
    outfile = os.path.join(dest_dir, basename+"_"+suffix+'.bvh')
    p = BVHParser()
    bvh = p.parse(filename, start=startframe, stop=endframe)
    if bvh.values.values.shape\[0\]>=(endframe-startframe):
        writer = BVHWriter()
        with open(outfile,'w') as f:
            writer.write(bvh, f)
    else:
        print("EOF REACHED")


/nfs/deepwave/home/simonal/dev/ListenDenoiseAction_flowmatching/MoGlow_FPP/utils/cut_wav.py