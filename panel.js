import { GUI } from './three.js/examples/jsm/libs/lil-gui.module.min.js';

function refresh_scene (scene, meshOpe){
    scene.remove(scene.children[scene.children.length - 1]); 
    meshOpe.display(scene);
}

var SingletonPanel = {

    gui : new GUI(),
    b_show_edges : true,
    b_show_verts : true,
    point_width : 1,
    init: function(scene, meshOpe) {
        const param = { 'show edges': true, 'show points': true, 'point width': 1. };
        SingletonPanel.gui.add( param, 'show edges' ).onChange( function ( val ) {
            SingletonPanel.b_show_edges = val;
            refresh_scene(scene, meshOpe);
        } );
        SingletonPanel.gui.add( param, 'show points' ).onChange( function ( val ) {
            SingletonPanel.b_show_verts = val;
            refresh_scene(scene, meshOpe);
        } );
        SingletonPanel.gui.add( param, 'point width', 0.1, 2, 0.1 ).onChange( function ( val ) {
            SingletonPanel.point_width = val;
            refresh_scene(scene, meshOpe);
        } );
    }
};

export{ SingletonPanel };