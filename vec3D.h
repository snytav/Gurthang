/*
 * vec3D.h
 *
 *  Created on: Jun 1, 2018
 *      Author: snytav
 */

#ifndef VEC3D_H_
#define VEC3D_H_

class vec3d{
public:
	double x,y,z;

	vec3d(){x = 0.0;y = 0.0;z = 0.0;}

	vec3d(double x1,double y1,double z1){x = x1;y = y1;z = z1;}
	void Set(double x1,double y1,double z1){x = x1;y = y1;z = z1;}
	vec3d & operator=(double3 d){x = d.x;y = d.y;z = d.z; return *this;}
	vec3d & operator=(vec3d & d){x = d.x;y = d.y;z = d.z; return *this;}
	vec3d & operator+(vec3d & d){vec3d p=*this; p.x += d.x;p.y += d.y;p.z += d.z; return p;}
//	vec3d & operator*(double  d){x *= d;y *= d.y;z *= d;}



	~vec3d(){}
};


#endif /* VEC3D_H_ */
