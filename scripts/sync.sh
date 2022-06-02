# Sync remote with local. Don't actually use this script!
# c+p the below line!
rsync -a --progress --exclude '*/checkpoints/*' cedar:'/home/taodav/scratch/uncertainty/results/uf8*' ./