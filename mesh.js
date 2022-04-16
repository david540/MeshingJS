
import * as THREE from 'three';
import { SingletonPanel } from './panel.js'

class MeshOpe {
    constructor( mesh ){
        this.reset(mesh);
	}
    display(scene){
        const cp_mesh = this.mesh.clone();
        scene.add( cp_mesh );
        if(SingletonPanel.b_show_edges) cp_mesh.add( this.wireframe );
        if(SingletonPanel.b_show_verts){
            this.points.material.size = SingletonPanel.point_width;
            cp_mesh.add( this.points );
        }
    }
    reset(mesh){
        if (typeof this.mesh !== 'undefined' && this.mesh !== null) delete(this.mesh);
        this.mesh = mesh;
        this.wireframe = new THREE.LineSegments(new THREE.WireframeGeometry( mesh.geometry ), new THREE.LineBasicMaterial( { color: 0x000000, linewidth:2 } ));
        let pos = mesh.geometry.attributes.position;
        var colors = []; for(var i = 0; i < pos.count; i++) colors.push(255, 0, 0);
        const geom_pt = new THREE.BufferGeometry();
        geom_pt.setAttribute( 'position', pos ); geom_pt.setAttribute( 'color', new THREE.Float32BufferAttribute( colors, 3 ) );
        const sprite = new THREE.TextureLoader().load( './three.js/examples/textures/sprites/disc.png' );
        const mt_pt = new THREE.PointsMaterial( { vertexColors: true, size: SingletonPanel.point_width, sizeAttenuation: true, map: sprite, alphaTest: 0.5, transparent: true } );
        this.points = new THREE.Points( geom_pt, mt_pt );
    }
}

export{ MeshOpe };