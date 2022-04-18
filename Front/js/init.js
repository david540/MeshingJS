import * as THREE from 'three';
import { SingletonPanel } from './panel.js'
import { MeshOpe } from './mesh.js'
import { OBJLoader } from './three.js/examples/jsm/loaders/OBJLoader.js';

let container;
let camera, scene, renderer;

init();
animate();

function init() {



    

    // ground

    var object;
    var meshOpe;

    const manager = new THREE.LoadingManager();
    const loader = new OBJLoader( manager );

    loader.load( '../../examples/emerald.obj', function ( obj ) {
        object = obj;
        object.position.x = 0; object.position.y = 0; object.position.z = 0;
        const mesh = object.children[0].clone();
        meshOpe = new MeshOpe(mesh);
        SingletonPanel.init(scene, meshOpe);
        meshOpe.display(scene);
    });

    document.querySelector('.inputfile').addEventListener('change', function(e){
        var dae_path;
        var files = e.currentTarget.files;
        if(files.length < 1) return;
        var extraFiles = {}, file;
        for (var i = 0; i < 1; i++){//files.length; i++) {
          file = files[i];
          extraFiles[file.name] = file;
          dae_path = file.name;
        }
        const manager = new THREE.LoadingManager();
        manager.setURLModifier(function (url, path) {
          url = url.split('/');
          url = url[url.length - 1];
          if (extraFiles[url] !== undefined) {
            var blobURL = URL.createObjectURL(extraFiles[url]);
            console.log(blobURL); //Blob location created from files selected from file input
            return blobURL;
          }return url;
        });
        const loader = new OBJLoader( manager );
        loader.load(dae_path, function ( obj ) {
            object = obj;
            scene.remove(scene.children[scene.children.length - 1]);
            object.position.x = 0; object.position.y = 0; object.position.z = 0;
            const mesh = object.children[0].clone();
            meshOpe.reset(mesh);
            meshOpe.display(scene);
        });
    });

    
    
    window.addEventListener( 'resize', onWindowResize );
}



function animate() {
    requestAnimationFrame( animate );
    render();
}
function render() {
    renderer.render( scene, camera );
}
