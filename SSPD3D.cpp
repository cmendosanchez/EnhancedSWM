#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <math.h>
#include <sys/stat.h>
#include <unordered_map>
#include "dirent.h"
#include <chrono>
#include <omp.h>
#include <experimental/filesystem>
//g++ -std=c++14 -O3 SSPD3D.cpp -o SSPD3D -ffast-math -fopenmp -lstdc++fs

namespace fs = std::experimental::filesystem;
using namespace std;





char * str_to_char_array(string s){
    int length = s.length()+1;
    char * char_array = new char[length];
#pragma omp parallel for
    for (unsigned short int i = 0; i<=length;i++){
        char_array[i] = s[i];
    }
    return char_array;
}



void write_bundles(string subject_name, string output_path, vector<vector<float>> &assignment,vector<string> &names ,int ndata_fiber,
                   vector<float> &subject_data){
    int npoints = ndata_fiber/3;
    ofstream bundlesfile;
    struct stat sb;
    char * output_folder = str_to_char_array(output_path);
    if (stat(output_folder, &sb) == 0 && S_ISDIR(sb.st_mode)){
        char * command =  str_to_char_array("rm -r "+output_path);
        int del = system(command);
    }
    mkdir(output_folder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    for (unsigned int i = 0; i<assignment.size();i++){
        if (assignment[i].size()!=0){
            string bundlesdata_path = output_path+"/"+subject_name+"_to_"+names[i]+".bundlesdata";
            char * bundlesdata_file = str_to_char_array(bundlesdata_path);
            FILE *fp = fopen(bundlesdata_file, "wb"); 	// Opening and writing .bundlesdata file.
            if (fp == NULL) {fputs ("File error opening .bundlesdata file\n",stderr); exit (1);}
            for (unsigned int j = 0; j < assignment[i].size(); j ++) {
                int fiber_index = assignment[i][j];
                fwrite(&npoints, sizeof(uint32_t),1, fp);

                //cout << &subject_data[fiber_index*ndata_fiber] << endl;

                fwrite(&subject_data[fiber_index*ndata_fiber], sizeof(float), ndata_fiber, fp);
            }
            fclose(fp);
            bundlesfile.open( output_path+"/"+subject_name+"_to_"+names[i]+".bundles", ios::out);
            bundlesfile<< "attributes = {"<<endl
                       <<"    \'binary\' : 1,"<<endl
                       <<"    \'bundles\' : [ '"<<(names[i])<<"', 0 ]," << endl
                       <<"    \'byte_order\' : \'DCBA\',"<<endl
                       <<"    \'curves_count\' : "<<assignment[i].size()<<","<< endl
                       <<"    \'data_file_name\' : \'*.bundlesdata\',"<<endl
                       <<"    \'format\' : \'bundles_1.0\',"<<endl
                       <<"    \'space_dimension\' : 3"<<endl
                       <<"  }"<<endl;
            bundlesfile.close();
            delete(bundlesdata_file);
        }
    }
    delete(output_folder);
}



vector<float> read_bundles(string path, unsigned short int ndata_fiber) {
    vector<float> data;
    char path2[path.length()+1];
    strncpy(path2, path.c_str(), sizeof(path2));
    path2[sizeof(path2) - 1] = 0;
    FILE *fp = fopen(path2, "rb");
	 // Open subject file.
    if (fp == NULL) {fputs ("File error opening file\n",stderr); exit (1);}
    fseek (fp, 0 , SEEK_END);
    long lSize = ftell(fp);                                // Get file size.
    unsigned int sfiber = sizeof(uint32_t) + ndata_fiber*sizeof(float); // Size of a fiber (bytes).  // Add 1 element (uint32_t) because in .bundles/.bundlesdata format the first element of each fiber/centroid corresponds to the amount of points in the fiber/centroid. In this case that number should be always the same.
    float buffer [sfiber];
    unsigned int nFibers = lSize/(float)sfiber;                 // Number of fibers
    rewind(fp);
    for(unsigned int j = 0; j < (nFibers); ++j)    // Copy fibers.
    {
        int r = fread(buffer, sizeof(float), (ndata_fiber+1), fp);     // Skip the first element of each fiber/centroid (number of points).;
        if (r == -1)
            cout<<"error reading buffer data";
        for(int s = 1; s < ndata_fiber+1; ++s)
        {
            data.push_back(buffer[s]);
        }
    }

    fclose(fp);
    return data;
}


float euclidean_distance(float x1, float y1, float z1, float x2, float y2, float z2){
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
}


vector<vector<float>> eucl_dist_traj(vector<float>t1,vector<float>t2,unsigned short int ndata_fiber,unsigned short int n_points){
    vector<vector<float>> matrix(n_points,vector<float> (n_points));
    unsigned int p_1 = 0;
    
    for(unsigned int i = 0;i<ndata_fiber;i+=3){
    	float x_1 = t1[i];
	float y_1 = t1[i+1];
	float z_1 = t1[i+2];
	unsigned int p_2 = 0;
	for(unsigned int j = 0; j<ndata_fiber;j+=3){
		float x_2 = t2[j];
		float y_2 = t2[j+1];
		float z_2 = t2[j+2];
		matrix[p_1][p_2] = euclidean_distance(x_1,y_1,z_1,x_2,y_2,z_2);
		p_2+=1;
	}
	p_1+=1;

}
    return matrix;
}

vector<float> t_dist(vector<float>t,unsigned short int ndata_fiber,unsigned short int n_points){
    vector<float> t_vector;
    for(unsigned int i = 0;i<ndata_fiber-3;i+=3){
    float x_1 = t[i];
    float y_1 = t[i+1];
    float z_1 = t[i+2];
    float x_2 = t[i+3];
    float y_2 = t[i+4];
    float z_2 = t[i+5];
    t_vector.push_back(euclidean_distance(x_1,y_1,z_1,x_2,y_2,z_2));
}
    return t_vector;
}

float point_to_segment(vector<float> p,vector<float> s1, vector<float> s2, float dps1,float dps2,float ds){
    float px = p[0];
    float py = p[1];
    float pz = p[2];
    float p1x = s1[0];
    float p1y = s1[1];
    float p1z = s1[2];
    float p2x = s2[0];
    float p2y = s2[1];
    float p2z = s2[2];
    float dpl;
    
    if(p1x==p2x && p1y==p2y && p1z==p2z){
	dpl = dps1;
    }
    else{
        float segl = ds;
	float x_diff = p2x-p1x;
        float y_diff = p2y-p1y;
        float z_diff = p2z-p1z;
	float u1 = ((px - p1x) * x_diff) + ((py - p1y) * y_diff) +((pz - p1z )*z_diff );
	float u = u1 / (segl * segl);

	if(u<0.00001 || u>1){
		dpl = min(dps1,dps2);
	
	}

	else{
		float ix = p1x + u*x_diff;
		float iy = p1y + u*y_diff;
		float iz = p1z + u*z_diff;
		dpl = euclidean_distance(px,py,pz,ix,iy,iz);
	}


    }
     return dpl;
}

float point_to_trajectory(vector<float> p,vector<float> t,vector<float> mdist_p,vector<float> t_dist,unsigned short int l_t){
	vector<float> p_to_seg_distances;
	//cout<<"point_to_trajectory\n"<<endl;
	for(unsigned short int i=0;i<l_t-1;i++){
		//cout<<i<<" ";
		vector<float> s1 = {t.begin()+3*i, t.begin() + 3*(i+1)};
		vector<float> s2 = {t.begin()+3*(i+1), t.begin() + 3*(i+2)};
		float p_to_seg_dist = point_to_segment(p,s1,s2,mdist_p[i],mdist_p[i+1],t_dist[i]);
		//cout<< p_to_seg_dist<<" ";
		p_to_seg_distances.push_back(p_to_seg_dist);
	}
	//cout<<*min_element(p_to_seg_distances.begin(), p_to_seg_distances.end())<<endl;
	return *min_element(p_to_seg_distances.begin(), p_to_seg_distances.end());
	//return 0.1;
}



float e_spd(vector<float> t1,vector<float>t2,vector<vector<float>> mdist, vector<float> t2_dist,unsigned short int ndata_fiber,unsigned short int n_points){
	vector<float> pt_distances;
	unsigned short int j = 0;
	float val =0;
	for(unsigned short int i = 0;i<ndata_fiber;i+=3){
		float x = t1[i];
		float y = t1[i+1];
		float z = t1[i+2];
		vector<float> p;
		p.push_back(x);
		p.push_back(y);
		p.push_back(z);
		float point_trajectory_distance = point_to_trajectory(p,t2,mdist[j],t2_dist,n_points);
		val+=point_trajectory_distance;
		pt_distances.push_back(point_trajectory_distance);
		j++;
	}
	float spd = val/j;
	//cout<<"spd dist: "<<spd<<endl;
	return spd;
}


float e_sspd(vector<float> t1,vector<float>t2,unsigned short int ndata_fiber,unsigned short int n_points){
	float val1;
	float val2;
	vector<float> t1_dist;
    	vector<float> t2_dist;
    	t1_dist = t_dist(t1,ndata_fiber,n_points);
    	t2_dist = t_dist(t2,ndata_fiber,n_points);
	vector<vector<float>> mdist_1;
	vector<vector<float>> mdist_2;
   	mdist_1 = eucl_dist_traj(t1,t2,ndata_fiber,n_points);
	mdist_2 = eucl_dist_traj(t2,t1,ndata_fiber,n_points);
	val1 = e_spd(t1,t2, mdist_1, t2_dist,ndata_fiber,n_points);
	val2 = e_spd(t2,t1, mdist_2, t1_dist,ndata_fiber,n_points);
	//cout<<"spd1 spd2: "<<val1<<" "<<val2<<" sspd: "<<(val1+val2)/2<<endl;
	return (val1+val2)/2;
}



int main(int argc, char *argv[])
{
    auto start = chrono::high_resolution_clock::now();
    unsigned short int n_points = atoi(argv[1]);
    string bundle_path = argv[2];
    string write_path = argv[3];
    unsigned short int ndata_fiber = n_points*3;
    vector<float> bundle_data;
    unsigned int nfibers_bundle;
    
    bundle_data = read_bundles(bundle_path+"data", ndata_fiber);
    nfibers_bundle = bundle_data.size()/ndata_fiber;
    //cout<<"Number of fibers: "<<nfibers_bundle<<endl;
    //for(unsigned i=0;i<ndata_fiber;i+=3){
	//cout<<bundle_data[i]<<" "<<bundle_data[i+1]<<" "<<bundle_data[i+2]<<"\n"; 
    //}
    
    
    vector<float> fiber_1 = {bundle_data.begin(), bundle_data.begin() + ndata_fiber};
    vector<float> fiber_2 = {bundle_data.begin()+ndata_fiber, bundle_data.begin() + ndata_fiber*2};  
    //cout<<endl;
    /*
     int count=0;
    for(unsigned int i=0;i<ndata_fiber;i+=3){
	cout<<fiber_1[i]<<" "<<fiber_1[i+1]<<" "<<fiber_1[i+2]<<"\n"; 
	count++;
    }
     cout<<count<<endl;
     cout<<endl;
    for(unsigned int i=0;i<ndata_fiber;i+=3){
	cout<<fiber_2[i]<<" "<<fiber_2[i+1]<<" "<<fiber_2[i+2]<<"\n"; 
    }*/
    vector<vector<float>> mdist;
    
    mdist = eucl_dist_traj(fiber_1,fiber_2,ndata_fiber,n_points);
    /*    
    for(unsigned int i=0;i<n_points;i++){
    for(unsigned int j=0;j<n_points;j++){   
    cout<<mdist[i][j]<<" ";}  
    cout<<endl;}*/
    vector<float> t1_dist;
    vector<float> t2_dist;
    t1_dist = t_dist(fiber_1,ndata_fiber,n_points);
    t2_dist = t_dist(fiber_2,ndata_fiber,n_points);
    //for(unsigned int i=0;i<n_points-1;i++){cout<<t2_dist[i]<<" ";}
    
    //cout<<"\n"<<endl;
    e_spd(fiber_1,fiber_2, mdist, t2_dist,ndata_fiber,n_points);

    vector<vector<float>> mdist_2;
    
    mdist_2 = eucl_dist_traj(fiber_2,fiber_1,ndata_fiber,n_points);
    e_spd(fiber_2,fiber_1, mdist_2, t1_dist,ndata_fiber,n_points);
    
    float test1 = e_sspd(fiber_1,fiber_2,ndata_fiber,n_points);
    float test2 = e_sspd(fiber_2,fiber_1,ndata_fiber,n_points);
    //cout<<"test1: "<<test1<<endl;
    //cout<<"test2: "<<test2<<endl;
    
    vector<vector<float>> distance_matrix(nfibers_bundle,vector<float> (nfibers_bundle,0.0));
    float* result=new float[nfibers_bundle];
    std::ofstream outFile(write_path);
    unsigned int nunProc = omp_get_num_procs();
    omp_set_num_threads(nunProc);
    //cout<<"n threads: "<<nunProc<<endl;
    #pragma omp parallel
    {
    
    #pragma omp for schedule(static,8) 	
    for(int i=0;i<nfibers_bundle;i++)
    {
        //cout << i<<endl;
	
	vector<float> f_1 = {bundle_data.begin()+ndata_fiber*i, bundle_data.begin() + ndata_fiber*(i+1)};
	for(int j=i;j<nfibers_bundle;j++){
	//if ()
	//vector<float> f_1 = {bundle_data.begin(), bundle_data.begin() + 63};
    	vector<float> f_2 = {bundle_data.begin()+ndata_fiber*j, bundle_data.begin() + ndata_fiber*(j+1)}; 
	float val = e_sspd(f_1,f_2,ndata_fiber,n_points);
        distance_matrix[i][j] = val;
    }
    }
    }

   for(unsigned short int i =0;i<nfibers_bundle;i++){
	for(unsigned short int j =i;j<nfibers_bundle;j++){
		distance_matrix[j][i] = distance_matrix[i][j];
}
}

   for(int i=0;i<nfibers_bundle;i++){
   for (const auto &e : distance_matrix[i]) outFile << e << " ";
   	outFile << '\n';
    }

    


    auto finish = chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::seconds> (finish - start).count();
    //cout<<"Time: "<<d<<" [s]"<<endl;

    return 0;
}

