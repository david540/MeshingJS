
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import {mergeVertices} from './GeometryUtils.js';
import * as interfaceUtils from './interfaceUtils.js';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';

class Mesh {
    constructor(path = './examples/emerald.obj') {
        this.filePath = path
        this.object = new THREE.Group()
        this.pointGroup = new THREE.Group()
        this.mesh = null
        this.computation_mesh = null
        this.scale_val = 1.
        this.translates = {x : 0, y : 0, z : 0}
        this.wireframe = null
        this.pointsCloud = null
    }
    recenter_and_rescale(obj){
       // obj.translateX(this.translates.x)
       // obj.translateY(this.translates.y)
       // obj.translateZ(this.translates.z)
        obj.scale.set(10./this.scale_val, 10./this.scale_val, 10./this.scale_val);
    }

    async loadMesh(camera) {
        //if(this.object) camera.remove(this.object)
        camera.position.fromArray([0, 0, 0]);
        
        this.object = new THREE.Group()
        //camera.add(this.object);
        //this.object.position.set(0,0,-20);
        //Load object
        const loadingManager = new THREE.LoadingManager();
        const objLoader = new OBJLoader(loadingManager);
        await objLoader.load(this.filePath, ((obj) => {
            //obj.position.fromArray([0, 0, 0]);
            obj.name = "mesh"
            this.mesh = obj.children[0]
            this.mesh.geometry.computeBoundingBox();
            
            let bbox = this.mesh.geometry.boundingBox;
            this.translates = {x : -bbox.min.x, y : -bbox.min.y, z : -bbox.min.z}
            let max_scale = Math.max(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y);
            this.scale_val = Math.max(bbox.max.z - bbox.min.z, max_scale);
            this.recenter_and_rescale(this.mesh);
            
            this.computation_mesh = this.mesh.clone()
            this.computation_mesh.geometry = mergeVertices(this.mesh.geometry, 1e-8)

            
            this.object.position.copy( camera.position );
            //this.object.translateZ( - 10 );
            this.object.add(this.mesh);

            this.wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(this.mesh.geometry), new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 }));
            this.wireframe.name = "wireframe"
            this.recenter_and_rescale(this.wireframe)
            this.object.add(this.wireframe)

            this.computePointsCloud()
        }).bind(this));

    }

    computePointsCloud() {
        //const loader = new THREE.TextureLoader();
       // const texture = loader.load( '../examples/disc.png' );
        /*
        const pointsMaterial = new THREE.PointsMaterial( {
            color: new THREE.Color( 255, 0, 0 ),
            map: texture,
            size: 0.01,
            alphaTest: 0.5
        } );*/
        const geometry = new THREE.BufferGeometry();
        const positionAttribute = this.computation_mesh.geometry.getAttribute( 'position' );

        const colors = [];
        const sizes = [];
        const singu_valences = [];
        const color = new THREE.Color();
        for ( let i = 0, l = positionAttribute.count; i < l; i ++ ) {
            color.setHSL( 0.01 + 0.1 * ( i / l ), 1.0, 0.5 );
            color.toArray( colors, i * 3 );
            sizes[ i ] = 10. / Math.sqrt(positionAttribute.count);
            singu_valences[i] = 4;
        }
        geometry.setAttribute( 'position', positionAttribute );
        geometry.setAttribute( 'customColor', new THREE.Float32BufferAttribute( colors, 3 ) );
        geometry.setAttribute( 'size', new THREE.Float32BufferAttribute( sizes, 1 ) );
        geometry.setAttribute( 'singu_valence', new THREE.Int32BufferAttribute( singu_valences, 1 ) );

        
        const material = new THREE.ShaderMaterial( {
            uniforms: {
                color: { value: new THREE.Color( 0xffffff ) },
                pointTexture: { value: new THREE.TextureLoader().load( '../examples/disc.png' ) },
                alphaTest: { value: 0.9 }
            },
            vertexShader: document.getElementById( 'vertexshader' ).textContent,
            fragmentShader: document.getElementById( 'fragmentshader' ).textContent

        } );
        
        this.pointsCloud = new THREE.Points(geometry, material)
        //this.pointsCloud.name = "pointsCloud";
        this.recenter_and_rescale(this.pointsCloud)
        this.pointGroup = new THREE.Group()
        this.pointGroup.add(this.pointsCloud)
        this.object.add(this.pointGroup);
    }

    updateDisplay(displayOptions) {
        this.pointsCloud.visible = displayOptions.bIsShowVerts
        this.wireframe.visible = displayOptions.bIsShowEdges
        this.pointsCloud.material.size = displayOptions.pointWidth
    }

    extractVertices() {
        if(!this.mesh) return []
        return interfaceUtils.CreateVectorDouble(this.computation_mesh.geometry.getAttribute("position").array);
    }

    extractFaceIndices() {
        if(!this.mesh) return []
        return interfaceUtils.CreateVectorInt(this.computation_mesh.geometry.index.array);
    }
    computeFF() {
        let data = interfaceUtils.ExtractArray(
            Module.computeFF(this.extractFaceIndices(), this.extractVertices(), 
            interfaceUtils.CreateVectorInt(this.pointsCloud.geometry.attributes.singu_valence.array)));
        
        const nh = data[0]
        const nv = data[1]
        const dir_arr = data.slice(2, 2 + nh * 2).concat(data.slice(2, 2 + nh * 2).map(function(x) {return -x}));
        const vert_arr = data.slice(2 + nh * 2, 2 + nh * 4).concat(data.slice(2 + nh * 2, 2 + nh * 4))
        const singu_arr = data.slice(2 + nh * 4, 2 + nh * 4 + nv)
        this.pointsCloud.geometry.setAttribute( 'singu_valence', new THREE.Int32BufferAttribute( singu_arr, 1 ) );
        //const attr = this.mesh.pointsCloud.geometry.attributes;
        //  attr.singu_valence.array[ this.selVert ] --;
        this.pointsCloud.geometry.attributes.singu_valence.needsUpdate = true;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array(vert_arr), 3 ) );
        geometry.setAttribute( 'normal', new THREE.BufferAttribute( new Float32Array(dir_arr), 3 ) );
        const material = new THREE.MeshBasicMaterial( { color: 0x0000ff } );
        const mesh = new THREE.Mesh( geometry, material )
        this.recenter_and_rescale(mesh)
        let helper = new VertexNormalsHelper( mesh, 0.03, new THREE.Color(255, 0, 0), 3 );
        helper.name = "frameField"
        this.object.add(helper)

    }
    computeParam() {
        const texture = new THREE.TextureLoader().load( "examples/quad_param.png" );
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.repeat.set( 1, 1 );
        
        let uvs = interfaceUtils.ExtractArray(Module.compute_param(this.extractFaceIndices(), this.extractVertices(), 
                                    interfaceUtils.CreateVectorInt(this.pointsCloud.geometry.attributes.singu_valence.array)));
        this.mesh.geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
        this.mesh.material = new THREE.MeshLambertMaterial( {  
            transparent: true,
            map: texture
        } );
    }
}

export { Mesh };
