import os
import sys
from subprocess import call
import shutil
import argparse
from fiber_bundle_filters import matrix_dist_filtering, Consistency,ConsistencyValsFiltering, ConvexHullFiltering,distance_matrix_Dend, distance_matrix_SSPD
import numpy as np
import read_write_bundle as bt
import time
from joblib import Parallel, delayed
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import length
import pickle
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram, save_trk, save_vtk_streamlines
import math
import nibabel as nb
import nibabel.streamlines.tck as TF
from nibabel.cmdline import tck2trk
from dipy.io.streamline import load_tractogram
from numpy.linalg import inv
from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines import Field
import random
T=np.load('tck2bundles.npy')

def savetck(fibs,outname):
	centroids_tractogram_file = Tractogram(streamlines = fibs)
	centroids_tractogram_file.affine_to_rasmm = np.eye(4)
	centroids_tck = TF.TckFile(centroids_tractogram_file,header = {'timestamp':0})
	centroids_tck.save(outname)
	#bundle = StatefulTractogram(fibs, 'MNI152_T1_1mm.nii.gz', Space.VOX)
	#save_tractogram(bundle, outname,bbox_valid_check=False)

def savetrk(fib,outname):
	nib.streamlines.save(fib, outname) 

def trk2bundles(trkfile,bundlefile):
	fibtrk,headtrk=nb.streamlines.load(trkfile)
	nb.streamlines.save(bundlefile)


def convert_folder(folder,extension,nthreads):
	bundles =  [f for f in os.listdir(folder) if f.endswith('.bundles')]
	if not os.path.exists(folder+'_'+extension):
		os.mkdir(folder+'_'+extension)
		
	if extension == 'tck':
		all_bun = []
		for bun in bundles:
			try:
				fibs = bt.read_bundle(folder+'/'+bun)
				fibs_aff = apply_aff_bundle_parallel(fibs,'',inv(T),nthreads)
				savetck(fibs_aff,folder+'_'+extension+'/'+bun.split('.')[0]+'.tck')
				all_bun+=fibs_aff
			except:
				continue
		#savetck(all_bun,'all_bun.tck')

	elif extension == 'trk':
		if not os.path.exists(folder+'_tck'):
			os.mkdir(folder+'_tck')
		all_bun = []
		for bun in bundles:
			try:
				fibs = bt.read_bundle(folder+'/'+bun)
				fibs_aff = apply_aff_bundle_parallel(fibs,'',inv(T),nthreads)
				all_bun+=fibs_aff
				savetck(fibs_aff,folder+'_tck/'+bun.split('.')[0]+'.tck')
				tractogram_tck = load_tractogram(folder+'_tck/'+bun.split('.')[0]+'.tck','MNI152_T1_1mm.nii.gz',bbox_valid_check=False)
				tractogram_trk = save_tractogram(tractogram_tck,folder+'_'+extension+'/'+bun.split('.')[0]+'.trk',bbox_valid_check=False)
			except:
				continue
		#savetck(all_bun,'all_bun.tck')
		#all_bun_tck = load_tractogram('all_bun.tck','MNI152_T1_1mm.nii.gz',bbox_valid_check=False)
		#save_tractogram(all_bun_tck,'all_bun.trk',bbox_valid_check=False)

def folder2bundles(folder,extension,nthreads):
	bundles =  [f for f in os.listdir(folder) if f.endswith(extension)]
	if not os.path.exists(folder+'_'+extension+'2bundles'):
		os.mkdir(folder+'_'+extension+'2bundles')
		
	if extension == 'tck':
		for bun in bundles:
			fibs_tck = load_tractogram(folder+'/'+bun,'MNI152_T1_1mm.nii.gz').streamlines
			fibs_aff = apply_aff_bundle_parallel(fibs_tck,'',T,nthreads)
			bt.write_bundle(folder+'_'+extension+'2bundles/'+bun.split('.')[0]+'.bundles',fibs_aff)

	
	elif extension == 'trk':
		if not os.path.exists(folder+'_trk2tck'):
			os.mkdir(folder+'_trk2tck')

		for bun in bundles:
			fibs_trk = load_tractogram(folder+'/'+bun,'same')
			save_tractogram(fibs_trk, folder+'_trk2tck/'+bun.split('.')[0]+'.tck')
			#fibs_aff = apply_aff_bundle_parallel(fibs_tck,'',T)
			fibs_tck = load_tractogram(folder+'_trk2tck/'+bun.split('.')[0]+'.tck','MNI152_T1_1mm.nii.gz').streamlines
			fibs_aff = apply_aff_bundle_parallel(fibs_tck,'',T,nthreads)
			bt.write_bundle(folder+'_'+extension+'2bundles/'+bun.split('.')[0]+'.bundles',fibs_aff)
			#tractogram_tck = load_tractogram(folder+'_tck/'+bun.split('.')[0]+'.tck','MNI152_T1_1mm.nii.gz')
			#tractogram_trk = save_tractogram(tractogram_tck,folder+'_'+extension+'/'+bun.split('.')[0]+'.trk')
	
	
		
def apply_aff_point(inPoint,t):
	#Tfrm = N.array([[t[1,0], t[1,1], t[1,2], t[0,0]],[t[2,0], t[2,1], t[2,2], t[0,1]],[ t[3,0], t[3,1], t[3,2], t[0,2]],[0, 0, 0, 1]])
	Tfrm = t
	#Tfrm = N.array([[0.6, 0, 0, 0],[0, -0.6, 0, 5],[ 0, 0, -0.6, 0],[0, 0, 0, 1]])
	tmp = Tfrm * np.transpose(np.matrix(np.append(inPoint,1)))
	outpoint = np.squeeze(np.asarray(tmp))[0:3]
	return outpoint


def apply_aff_fiber(f,t):
	#print(idx)
	newfib=[]
	for p in f:
		pt=apply_aff_point(p,t)
		newfib.append((pt))
	return np.asarray(newfib,dtype=np.float32)

def apply_aff_bundle(bunIn,bunOut,t):
	#points=BT.read_bundle(bunIn)
	#points = BT.read_bundle(bunIn)
	points = bunIn
	#print(len(points[0]))
	newPoints=[]
	for idx,fib in enumerate(points):
		print(idx)
		newfib=[]
		for p in fib:
			pt=apply_aff_point(p,t)
			newfib.append((pt))
		newPoints.append(np.asarray(newfib,dtype=np.float32))
	#BT.write_bundle(bunOut, newPoints)
	return newPoints

def apply_aff_bundle_parallel(bunIn,bunOut,t,nthreads):
	points = bunIn
	newPoints = Parallel(n_jobs=nthreads)(delayed(apply_aff_fiber)(f,t) for f in bunIn)
	return newPoints


def dME(fib_i,fib_j):
    dist_1 = np.linalg.norm(fib_i-fib_j,axis=1)
    dist_2 = np.linalg.norm(fib_i-np.flip(fib_j,axis=0),axis=1)
    dist = min(max(dist_1),max(dist_2))
    return dist

def dNE(fib_i,fib_j):
    dist_1 = np.linalg.norm(fib_i-fib_j,axis=1)
    dist_2 = np.linalg.norm(fib_i-np.flip(fib_j,axis=0),axis=1)
    dist = min(max(dist_1),max(dist_2))
    term1 = list(length([fib_i]))[0]
    term2 = list(length([fib_j]))[0]
    factor = (abs(term1-term2)/max(term1,term2))+1
    NT = (factor*factor)-1
    return dist+NT

def create_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)



def apply_SSPD_filter(results_dir,out_dir_idx,seg_b,selec_fil,filter_idx,bundle,out_bun,write_indices,p1,p2):
	try:
		if seg_b!=None:
			param1 = int(np.load('param_1/'+selec_fil+'_'+bundle_id[bundles_each_class[seg_b.split('subject_to_')[1]]]+'.npy'))
			param2 = int(np.load('param_2/'+selec_fil+'_'+bundle_id[bundles_each_class[seg_b.split('subject_to_')[1]]]+'.npy'))
			#print(seg_b,param1,param2,bundles_each_class[seg_b.split('subject_to_')[1]])
		else:
			param1 = p1
			param2 = p2
			#print(seg_b,p1,p2)

		fibers = bt.read_bundle(bundle)
		matrix_dist_SSPD = distance_matrix_SSPD(bundle)
		filtered_bundle,noise_idx = matrix_dist_filtering(matrix_dist_SSPD,param1,param2,fibers)
		bt.write_bundle(out_bun,filtered_bundle)

		if write_indices==1:
			with open(results_dir+'_idx/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt') as f:
				lines = f.readlines()
			lines_edit = []
			for l in lines:
				lines_edit.append(l.split('.')[0])

			
			idx_seg = lines_edit.copy()
			indexes = noise_idx
			for index in sorted(indexes, reverse=True):
				del idx_seg[index]

			with open(out_dir_idx+'/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt', 'w') as fp:
				for item in idx_seg:
					# write each item on a new line
					fp.write("%s\n" % item)
			fp.close()
	except:
		return

def apply_filter_parallel(results_dir,out_dir_idx,seg_b,selec_fil,filter_idx,bundle,out_bun,write_indices,p1,p2):
	try:
		if seg_b!=None:
			param1 = int(np.load('param_1/'+selec_fil+'_'+bundle_id[bundles_each_class[seg_b.split('subject_to_')[1]]]+'.npy'))
			param2 = int(np.load('param_2/'+selec_fil+'_'+bundle_id[bundles_each_class[seg_b.split('subject_to_')[1]]]+'.npy'))
			#print(seg_b,param1,param2,bundles_each_class[seg_b.split('subject_to_')[1]])
		else:
			param1 = p1
			param2 = p2
			#print(seg_b,p1,p2)

		if filter_idx == '1':
			fibers = bt.read_bundle(bundle)
			matrix_dist_Dend = distance_matrix_Dend(fibers)
			filtered_bundle,noise_idx = matrix_dist_filtering(matrix_dist_Dend,param1,param2,fibers)
			bt.write_bundle(out_bun,filtered_bundle)

		elif filter_idx == '3':
			fibers = bt.read_bundle(bundle)
			dist_matrix_MDF = bundles_distances_mdf(fibers,fibers)
			fibers_mean_consistency = Consistency(bundle,dist_matrix_MDF,8,param2) 
			filtered_bundle, noise_idx = ConsistencyValsFiltering(fibers,fibers_mean_consistency,param1)
			bt.write_bundle(out_bun,filtered_bundle)

		elif filter_idx=='4':
			filtered_bundle,noise_idx = ConvexHullFiltering(bundle,param1,param2,21)
			#print(filtered_bundle)
			bt.write_bundle(out_bun,filtered_bundle)

		if write_indices==1:
			with open(results_dir+'_idx/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt') as f:
				lines = f.readlines()
			lines_edit = []
			for l in lines:
				lines_edit.append(l.split('.')[0])
			#print(lines_edit)

			idx_seg = lines_edit.copy()
			indexes = noise_idx
			for index in sorted(indexes, reverse=True):
				del idx_seg[index]

			with open(out_dir_idx+'/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt', 'w') as fp:
				for item in idx_seg:
					# write each item on a new line
					fp.write("%s\n" % item)
			fp.close()
		
	except Exception as e:
		print("Exception",seg_b,e)





def apply_MFF(results_dir,out_dir_idx,seg_b,bun,cent,threshold,out_fascicle):
	try:
		fibs_distances = []
		bundle = bt.read_bundle(bun)
		centroid = bt.read_bundle(cent)
		for sl in bundle:
			dNE_dist = dNE(sl,centroid[0]) 
			fibs_distances.append(dNE_dist)
		MFF_idx = np.where(np.asarray(fibs_distances) <= threshold)[0]
		noise_idx = np.where(np.asarray(fibs_distances) > threshold)[0]
		#print(MFF_idx)
		main_fiber_fascicle = [ bundle[i] for i in MFF_idx]
		if len(main_fiber_fascicle)!=0:
			bt.write_bundle(out_fascicle,main_fiber_fascicle)


			with open(results_dir+'_idx/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt') as f:
				lines = f.readlines()
			lines_edit = []
			for l in lines:
				lines_edit.append(l.split('.')[0])
			#print(lines_edit)
			idx_seg = lines_edit.copy()
			for index in sorted(noise_idx, reverse=True):
				del idx_seg[index]

			with open(out_dir_idx+'/'+seg_b.split('subject_to_')[1].split('.')[0]+'.txt', 'w') as fp:
				for item in idx_seg:
					# write each item on a new line
					fp.write("%s\n" % item)
			fp.close()

	except Exception as e:
		print("Exception:",seg_b,e)
		

bundles_each_class = {}
for i in range(0,4):
	with open('clusters_bundles/cluster_'+str(i)+'.txt') as f:
		lines = f.readlines()
		#print(lines)

	for bun in lines:
		bundles_each_class[bun.split('.')[0]+'.bundles'] = str(i)

bundle_id = {'0':'rh_PoC-PrC_0.bundles','1':'lh_IP-MT_1.bundles','2':'rh_IP-IT_1.bundles','3':'lh_CAC-PoCi_0.bundles'} 
Filters = {'1':'ConnectivityPattern','2':'SSPD','3':'Consistency','4':'ConvexHull'}


with open('MFF_thresholds.pkl', "rb") as input_file:
	b_thresholds = pickle.load(input_file)
	#print(b_thresholds)
	ths = list(b_thresholds.values())
	#print(np.mean(ths),np.std(ths))


def main():

	parser = argparse.ArgumentParser(description='Enhanced segmentation')
	parser.add_argument('--in_data',default=None, type=str, help='Input tractogram')
	parser.add_argument('--extension',default=None, type=str, help='Format of input tractogram')
	parser.add_argument('--in_folder',default=None, type=str, help='Input folder of bundles')
	parser.add_argument('--in_folder_extension',default=None, type=str, help='Format of bundles from folder')
	parser.add_argument('--out_dir', type=str, help='Output directory')
	parser.add_argument('--filter',default=0, type=str, help='Fiber bundle filter')
	parser.add_argument('--MFF',default=0, type=int, help='Apply main fiber fascicle identification')
	parser.add_argument('--p1',default=None, type=int, help='Parameter 1 of the fiber bundle filter')
	parser.add_argument('--p2',default=None, type=int, help='Parameter 2 of the fiber bundle filter')
	parser.add_argument('--nthreads',default=-1, type=int, help='Number of threads')
	args = parser.parse_args()

	nthreads = args.nthreads
	if args.in_data!= None:
		#extension = args.in_data.rsplit('.', 1)[-1]
		#extension = args.extension
		folders = args.out_dir.rsplit('/', 1)[0]
		#print(args.in_data,extension,args.out_dir)
		create_dirs(folders)
	

	#Apply fiber bundle segmentation based on multi-subject atlas
	if args.extension == 'tck' and args.in_data!= None:
		#print('.tck to .bundles')
		tractogram_tck = load_tractogram(args.in_data,'MNI152_T1_1mm.nii.gz')
		#tractogram_trk = save_tractogram(tractogram_tck,args.in_data.rsplit('.', 1)[0]+'.trk')
		#print(tractogram_tck.streamlines)
		fibs = []
		for fib in tractogram_tck.streamlines:
			fibs.append(fib)#(fibTCK[i])#[0]
		#init = time.time()
		#fibs_aff = apply_aff_bundle(fibs,'',T)
		#end = time.time()
		#print("Execution time of affine: "+str(round(end-init,2))+" seconds")
		init = time.time()
		fibs_aff = apply_aff_bundle_parallel(fibs,'',T,nthreads)
		end = time.time()
		#print("Execution time of affine: "+str(round(end-init,2))+" seconds")
		try:
			fibs_21p = set_number_of_points(fibs_aff,21)
		except:
			fibs_21p = []
			for f in fibs_aff:
				if len(f)>=2:
					f_21p = set_number_of_points([f],21)
					fibs_21p.append(f_21p[0])
		
		#aa = random.sample(fibs_21p, 10000)
		#bt.write_bundle('tract_tck2bundles.bundles',aa)

		bt.write_bundle(args.in_data.rsplit('.', 1)[0]+'.bundles',fibs_21p)

	if args.extension == 'trk' and args.in_data!= None:
		tractogram_trk = load_tractogram(args.in_data,'same')
		#print(tractogram_trk)
		save_tractogram(tractogram_trk, args.in_data.rsplit('.', 1)[0]+'.tck')
		tractogram_tck = load_tractogram(args.in_data.rsplit('.', 1)[0]+'.tck','MNI152_T1_1mm.nii.gz')
		fibs = []
		for fib in tractogram_tck.streamlines:
			fibs.append(fib)
		
		init = time.time()
		fibs_aff = apply_aff_bundle_parallel(fibs,'',T,nthreads)
		end = time.time()
		#print("Execution time of affine: "+str(round(end-init,2))+" seconds")
		try:
			fibs_21p = set_number_of_points(fibs_aff,21)
		except:
			fibs_21p = []
			for f in fibs_aff:
				if len(f)>=2:
					f_21p = set_number_of_points([f],21)
					fibs_21p.append(f_21p[0])

		#aa = random.sample(fibs_21p, 10000)
		#bt.write_bundle('tract_trk2tck2bundles.bundles',aa)

		bt.write_bundle(args.in_data.rsplit('.', 1)[0]+'.bundles',fibs_21p)
	



	if args.extension == 'bundles' and args.in_data!= None:
		fibs = bt.read_bundle(args.in_data)
		fibs_21p = set_number_of_points(fibs,21)
		bt.write_bundle(args.in_data.rsplit('.', 1)[0]+'_21p.bundles',fibs_21p)
		arg = ['./main','21',args.in_data.rsplit('.', 1)[0]+'_21p.bundles','subject','AtlasRo','AtlasRo/atlasInformation.txt',args.out_dir,args.out_dir+'_idx']
		call(arg)

	if args.extension != 'bundles' and args.in_data!= None:
		arg = ['./main','21',args.in_data.rsplit('.', 1)[0]+'.bundles','subject','AtlasRo','AtlasRo/atlasInformation.txt',args.out_dir,args.out_dir+'_idx']
		call(arg)
	#return
	#Apply a fiber bundle filter
	if args.filter!=0 and args.MFF == 0 and args.in_data!=None:

		selected_filter = Filters[args.filter]
		create_dirs(folders+'/Filtered_bundles_'+selected_filter)
		create_dirs(folders+'/Filtered_bundles_'+selected_filter+'_idx')
		segmented_bundles = [f for f in os.listdir(args.out_dir) if f.endswith('.bundles')]
		if 'subject_to_lh_PoCi-SF_1.bundles' in segmented_bundles:
			segmented_bundles.remove('subject_to_lh_PoCi-SF_1.bundles')
		'''
		init = time.time()
		for segmented_bundle in segmented_bundles:
			param1 = int(np.load('param_1/'+selected_filter+'_'+bundle_id[bundles_each_class[segmented_bundle.split('subject_to_')[1]]]+'.npy'))
			param2 = int(np.load('param_2/'+selected_filter+'_'+bundle_id[bundles_each_class[segmented_bundle.split('subject_to_')[1]]]+'.npy'))
			out_bun = folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle
			print(segmented_bundle,param1,param2,out_bun) 
			try:
				#print("Apply filter")
				apply_filter(args.filter,param1,param2,args.out_dir+'/'+segmented_bundle,out_bun)

			except Exception as e: 
				print("Could not filter bundle ", segmented_bundle,e)
				#continue
		end = time.time()		
		print("Execution time of filtering: "+str(round(end-init,2))+" seconds")
		'''

		init = time.time()
		if args.filter!='2':
			Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)(args.out_dir,folders+'/Filtered_bundles_'+selected_filter+'_idx',segmented_bundle,selected_filter,args.filter,args.out_dir+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,1,'','') for segmented_bundle in segmented_bundles)
		else:
			for segmented_bundle in segmented_bundles:
				apply_SSPD_filter(args.out_dir,folders+'/Filtered_bundles_'+selected_filter+'_idx',segmented_bundle,selected_filter,args.filter,args.out_dir+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,1,'','')
				
		end = time.time()
		print("Execution time of fiber bundle filtering: "+str(round(end-init,2))+" seconds")	

	elif args.filter!=0 and args.MFF == 1 and args.in_data!=None:		
		print("Applying main fiber fascicle identification and fiber bundle filtering")
		selected_filter = Filters[args.filter]
		create_dirs(folders+'/Filtered_bundles_'+selected_filter)
		create_dirs(folders+'/Filtered_bundles_'+selected_filter+'_idx')
		create_dirs(folders+'/MFF')
		create_dirs(folders+'/MFF_idx')
		create_dirs(folders+'/Filtered_MFF_'+selected_filter)
		create_dirs(folders+'/Filtered_MFF_'+selected_filter+'_idx')
		segmented_bundles = [f for f in os.listdir(args.out_dir) if f.endswith('.bundles')]
		if 'subject_to_lh_PoCi-SF_1.bundles' in segmented_bundles:
			segmented_bundles.remove('subject_to_lh_PoCi-SF_1.bundles')
		#Filtering of segmented bundles
		init = time.time()
		if args.filter!='2':
			Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)(args.out_dir,folders+'/Filtered_bundles_'+selected_filter+'_idx',segmented_bundle,selected_filter,args.filter,args.out_dir+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,1,'','') for segmented_bundle in segmented_bundles)
		else:
			for segmented_bundle in segmented_bundles:
				apply_SSPD_filter(args.out_dir,folders+'/Filtered_bundles_'+selected_filter+'_idx',segmented_bundle,selected_filter,args.filter,args.out_dir+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,1,'','')
				
		end = time.time()
		print("Execution time of fiber bundle filtering: "+str(round(end-init,2))+" seconds")
		
		#Main fiber fascicle identification
		init = time.time()
		Parallel(n_jobs=nthreads)(delayed(apply_MFF)(args.out_dir,folders+'/MFF_idx',segmented_bundle,args.out_dir+'/'+segmented_bundle,'centroids/'+segmented_bundle.split('subject_to_')[1],b_thresholds[segmented_bundle.split('subject_to_')[1]],folders+'/MFF/'+segmented_bundle) for segmented_bundle in segmented_bundles)
		end = time.time()
		print("Execution time of mff identification: "+str(round(end-init,2))+" seconds")
		MFF_bundles = [f for f in os.listdir(folders+'/MFF') if f.endswith('.bundles')]
		#Filtering of main fiber fascicles
		init = time.time()
		if args.filter!='2':
			Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)(folders+'/MFF',folders+'/Filtered_MFF_'+selected_filter+'_idx',MFF,selected_filter,args.filter,folders+'/MFF/'+MFF,folders+'/Filtered_MFF_'+selected_filter+'/'+MFF,1,'','') for MFF in MFF_bundles)
		else:
			for MFF in MFF_bundles:
				apply_SSPD_filter(folders+'/MFF',folders+'/Filtered_MFF_'+selected_filter+'_idx',MFF,selected_filter,args.filter,args.out_dir+'/'+MFF,folders+'/Filtered_MFF_'+selected_filter+'/'+MFF,1,'','')
		end = time.time()
		print("Execution time of mff filtering: "+str(round(end-init,2))+" seconds")

	if args.extension != 'bundles' and args.MFF==0 and args.in_data!=None:
		convert_folder(args.out_dir,args.extension,nthreads)
		convert_folder(folders+'/Filtered_bundles_'+selected_filter,args.extension,nthreads)
		
	elif args.extension != 'bundles' and args.MFF==1 and args.in_data!=None:
		convert_folder(args.out_dir,args.extension,nthreads)
		convert_folder(folders+'/Filtered_bundles_'+selected_filter,args.extension,nthreads)
		convert_folder(folders+'/MFF',args.extension,nthreads)
		convert_folder(folders+'/Filtered_MFF_'+selected_filter,args.extension,nthreads)

	
	if args.in_folder!=None and args.extension!=None:
		selected_filter = Filters[args.filter]
		folders = args.in_folder.rsplit('/', 1)[0]
		create_dirs(folders+'/Filtered_bundles_'+selected_filter)

		if args.extension == 'bundles':
			segmented_bundles = [f for f in os.listdir(args.in_folder) if f.endswith('.bundles')]
			if args.filter!='2':
				Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)('','',None,selected_filter,args.filter,args.in_folder+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2) for segmented_bundle in segmented_bundles)
			else:
				for segmented_bundle in segmented_bundles:
					apply_SSPD_filter('','',None,selected_filter,args.filter,args.in_folder+'/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2)
	
		elif args.extension== 'tck':
			folder2bundles(args.in_folder,args.extension,nthreads)
			segmented_bundles = [f for f in os.listdir(args.in_folder+'_'+args.extension+'2bundles') if f.endswith('.bundles')]
			
			if args.filter!='2':
				Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)('','',None,selected_filter,args.filter,args.in_folder+'_'+args.extension+'2bundles/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2) for segmented_bundle in segmented_bundles)
			else:
				for segmented_bundle in segmented_bundles:
					apply_SSPD_filter('','',None,selected_filter,args.filter,args.in_folder+'_'+args.extension+'2bundles/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2)
			convert_folder(folders+'/Filtered_bundles_'+selected_filter,args.extension,nthreads)

		elif args.extension== 'trk':
			folder2bundles(args.in_folder,args.extension,nthreads)
			segmented_bundles = [f for f in os.listdir(args.in_folder+'_'+args.extension+'2bundles') if f.endswith('.bundles')]
			
			if args.filter!='2':
				Parallel(n_jobs=nthreads)(delayed(apply_filter_parallel)('','',None,selected_filter,args.filter,args.in_folder+'_'+args.extension+'2bundles/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2) for segmented_bundle in segmented_bundles)
			else:
				for segmented_bundle in segmented_bundles:
					apply_SSPD_filter('','',None,selected_filter,args.filter,args.in_folder+'_'+args.extension+'2bundles/'+segmented_bundle,folders+'/Filtered_bundles_'+selected_filter+'/'+segmented_bundle,0,args.p1,args.p2)
			convert_folder(folders+'/Filtered_bundles_'+selected_filter,args.extension,nthreads)
		
if __name__ == '__main__':
    main()
