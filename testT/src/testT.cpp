//
// Created by jin on 4/1/19.
//

#include "SLAMbase.h"

#include <algorithm>
#include <iostream>
using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;


#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

int main( int argc, char** argv )
{
    // string inFileName;


    // inFileName = "/home/jin/Data/RV_Data/Translation/Y1/frm_0001.dat";
    // SR4kFRAME f1 = readSRFrame(inFileName) ;

    // inFileName = "/home/jin/Data/RV_Data/Translation/Y2/frm_0001.dat";
    // SR4kFRAME f2 = readSRFrame(inFileName) ;


    cv::Mat rgb1 = cv::imread( "/home/jin/Data/04_04_2019/cameraStill/color/1554395077.593656341.png");
    cv::Mat rgb2 = cv::imread( "/home/jin/Data/04_04_2019/cameraStill/color/1554395077.869748494.png");

    cv::Mat depth1 = cv::imread( "/home/jin/Data/04_04_2019/cameraStill/aligned_depth/1554395077.593656341.png", -1);
    cv::Mat depth2 = cv::imread( "/home/jin/Data/04_04_2019/cameraStill/aligned_depth/1554395077.869748494.png", -1);

    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create( "GridSIFT" );
    _descriptor = cv::DescriptorExtractor::create( "SIFT" );
    vector< cv::KeyPoint > kp1, kp2; 
    _detector->detect( rgb1, kp1 );
    _detector->detect( rgb2, kp2 );

    // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    // vector< cv::KeyPoint > kp1, kp2; 
    // detector->detect( rgb1, kp1 ); 
    // detector->detect( rgb2, kp2 );
    // cv::Mat desp1, desp2;
    // descriptor->compute( rgb1, kp1, desp1 );
    // descriptor->compute( rgb2, kp2, desp2 );

    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    
    // 可视化， 显示关键点
    cv::Mat imgShow;
    cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgShow );
    cv::imwrite( "./data/keypoints.png", imgShow );
    cv::waitKey(0); //暂停等待一个按键
   
    // 计算描述子
    cv::Mat desp1, desp2;
    _descriptor->compute( rgb1, kp1, desp1 );
    _descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches; 
    cv::FlannBasedMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "./data/matches.png", imgMatches );
    cv::waitKey( 0 );

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 4*minDis)
            goodMatches.push_back( matches[i] );
    }

    // 显示 good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow( "good matches", imgMatches );
    cv::imwrite( "./data/good_matches.png", imgMatches );
    cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数
    
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 315.40594482421875;
    C.cy = 244.33926391601562;
    C.fx = 614.8074340820312;
    C.fy = 614.5072021484375;
    C.scale = 1000.0;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;


        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );

        
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1, 50, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    cv::Mat R;
    cv::Rodrigues(rvec, R); // R is 3x3
    cv::Mat T = cv::Mat::eye(4,4,CV_64F);

    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));
    

    cout<<"T="<<T<<endl;


    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );


    vector<cv::Point3f> pts_inlier1;
    vector<cv::Point3f> pts_inlier2;

    double sumDist = 0;
    double sumError = 0;
    for (size_t i=0; i<inliers.rows; i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[goodMatches[ inliers.ptr<int>(i)[0] ].queryIdx].pt;
        cv::Point2f p2 = kp2[goodMatches[ inliers.ptr<int>(i)[0] ].trainIdx].pt;

        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1 == 0 || d2 ==0){
            continue;
        }

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );
        pts_inlier1.push_back( pd1 );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );
        pts_inlier2.push_back( pd2 );

        // cout << "pd1 "<<pd1 <<endl;
        // cout << "pd2 "<<pd2 <<endl;
        cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
        cv::Mat dstMat = T*ptMat;
        cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));
        cout << "projPd1 "<<projPd1 <<endl;

        // cout << "(pd1*T-pd2) "<< norm(projPd1-pd2)*100 <<"mm" <<endl;
        // cout << "(pd1-pd2) "<< norm(pd1-pd2)*100 <<"mm" <<endl;

        sumDist = sumDist + norm(pd1-pd2)*100;
        sumError = sumError + norm(projPd1-pd2)*100;

    }

    cout << "avg error "<< sumError/inliers.rows <<"mm" <<endl;
    cout << "avg dist "<< sumDist/inliers.rows  <<"mm" <<endl;

    pcl::PointCloud<pcl::PointXYZ> inliersPC1;
    inliersPC1.points.resize (pts_inlier1.size());
    pcl::PointCloud<pcl::PointXYZ> inliersPC2;
    inliersPC2.points.resize (pts_inlier2.size());
    for (size_t i=0; i<pts_inlier2.size(); i++) {


        inliersPC1.points[i].x = pts_inlier1[i].x;
        inliersPC1.points[i].y = pts_inlier1[i].y;
        inliersPC1.points[i].z = pts_inlier1[i].z;

        inliersPC2.points[i].x = pts_inlier2[i].x;
        inliersPC2.points[i].y = pts_inlier2[i].y;
        inliersPC2.points[i].z = pts_inlier2[i].z;
    }


    inliersPC1.height = 1;
    inliersPC1.width = inliersPC1.points.size();
    cout<<"point cloud size = "<<inliersPC1.points.size()<<endl;

    pcl::io::savePCDFile( "/home/jin/Desktop/inliersPC1.pcd", inliersPC1 );

    inliersPC2.height = 1;
    inliersPC2.width = inliersPC2.points.size();
    cout<<"point cloud size = "<<inliersPC2.points.size()<<endl;

    pcl::io::savePCDFile( "/home/jin/Desktop/inliersPC2.pcd", inliersPC2 );

/*=================================================================================================*/
    vector<cv::Point3f> fixPts;
    vector<cv::Point3f> mvingPts;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[goodMatches[ i ].queryIdx].pt;
        cv::Point2f p2 = kp2[goodMatches[ i ].trainIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1 == 0 || d2 ==0){
            continue;
        }
        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );
        fixPts.push_back( pd1 );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );
        mvingPts.push_back( pd2 );
    }

    cout<<"fixPts: "<<fixPts.size()<<endl;
    cv::Mat outM3by4;// = cv::Mat::zeros(3,4,CV_64F);
    cv::Mat inliers3d;
    cv::estimateAffine3D(fixPts, mvingPts, outM3by4, inliers3d, 3, 0.999);
    cv::Mat rmat = outM3by4(cv::Rect(0,0,3,3));
    cv::Mat rvecN;

    cv::Rodrigues(rmat,rvecN);

    cv::Mat tvecN = outM3by4(cv::Rect(3,0,1,3));


    cout<<"affine M = "<<outM3by4<<endl;
    cout<<"R="<<rvecN<<endl;
    cout<<"t="<<tvecN<<endl;
    cout<<"inliers3d: "<<inliers3d.rows<<endl;

    return 0;
}
