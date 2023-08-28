// Finding optimal spectral index and threshold for mapping impervious surface

var seq_veg = ee.List.sequence(0.1, 0.7, 0.01);
var seq_built = ee.List.sequence(-0.50, 0.40, 0.01);

//print(seq_veg.size().multiply(2))

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask);
}

var boundary = ee.FeatureCollection(cities.filter(ee.Filter.inList('city',[city])))
//Map.addLayer(boundary)

// Surface temperature re-analysis

var dataset = ee.ImageCollection('ECMWF/ERA5/DAILY')
                .select('mean_2m_air_temperature')
                .filter(ee.Filter.date('2019-03-01', '2019-11-30'));

var celsium = function(image) {
  
  var ST = image.expression(
    'ST-273.15', {
    'ST': image.select('mean_2m_air_temperature')
    }).rename('LST');
  
  return(image.addBands([ST]))
}

var dataset_bands = dataset.map(celsium).select(['LST'])

//print(dataset_bands)

//var dataset = ee.ImageCollection('NCEP_RE/surface_temp')
//                  .filter(ee.Filter.date('2019-01-01', '2019-12-31'))
//                  .select('air');
var ndvi_mod = ee.ImageCollection('MODIS/006/MOD13Q1')
                  .filter(ee.Filter.date('2019-03-01', '2019-11-30'))
                  .select('NDVI');
                  
var gs = ee.FeatureCollection([gs_mur,gs_apa,gs_chp,gs_msk,gs_pus,gs_kal,gs_kur,gs_ros,gs_nsb,gs_nak])

var surf_temp = 
    ui.Chart.image
        .series({
          imageCollection: dataset_bands,
          region: boundary,
          reducer: ee.Reducer.mean(),
          scale: 30000,
          xProperty: 'system:time_start'
        })
        .setSeriesNames(['LST'])
        .setOptions({
          title: 'Средние значения температуры поверхности ERA5 reanalysis',
          series: {
            0: {
              targetAxisIndex: 0 ,
              type: "line" ,
              lineWidth: 2 ,
              pointSize: 0 ,
              color: "red"
            },
            
            hAxis: {
              title: 'Дата',
              format: "MM" ,
              titleTextStyle: { italic: false, bold: true }
            },
            vAxes: {
              0: {
                title: "Температура поверхности, С",
                baseline: 0 ,
                titleTextStyle: { bold: true , color: 'red' } ,
                viewWindow: { 
                  min: -30,
                  max: 40}
              },
          },
          curveType: 'function'
}});
        
//print(surf_temp);


var em_name = 'landcover'
var maxError = 0.001
var union = function(vector){
  var roinames = ee.List(vector.aggregate_array(em_name)).distinct();
  
  var byClass = function(roiname){
    
    var tempFC = vector.filter(ee.Filter.eq(em_name, roiname));
    var unionFC = tempFC.union(maxError);
  
  return ee.Feature(unionFC.first()).set(em_name, roiname)
  }
  var unified = ee.FeatureCollection(roinames.map(byClass));
  return unified
}

var endmembers = union(
  ee.FeatureCollection(
    em_points
    .filter(ee.Filter.inList('city',[city]))
    )
    .filter(ee.Filter.inList('class',['training']))
    .filter('region != "NM"'));
    
var ndvi_16days =
    ui.Chart.image
        .series({
          imageCollection: ndvi_mod,
          region: gs.filter(ee.Filter.inList('city',[city])), //endmembers.filter(ee.Filter.inList('landcover',['DF'])),
          reducer: ee.Reducer.mean(),
          scale: 250,
          xProperty: 'system:time_start'
        })
        .setSeriesNames(['NDVI'])
        .setOptions({
          title: 'Средние значения NDVI (MODIS)',
          series: {
            0: {
              targetAxisIndex: 0 ,
              type: "line" ,
              lineWidth: 2 ,
              pointSize: 0 ,
              color: "red"
            },
            
            hAxis: {
              title: 'Дата',
              format: "MM" ,
              titleTextStyle: { italic: false, bold: true }
            },
            vAxes: {
              0: {
                title: "NDVI",
                baseline: 0 ,
                titleTextStyle: { bold: true , color: 'green' } ,
                viewWindow: { 
                  min: -1,
                  max: 1}
              },
          },
          curveType: 'function'
}});

//print(ndvi_16days);

//Old end-members for Murmansk

//var meanBSold = ee.List([0.064,0.083,0.087,0.114,0.168,0.178,0.171,0.191,0.175,0.135])
//var meanDFold = ee.List([0.028,0.050,0.026,0.074,0.236,0.268,0.272,0.281,0.133,0.058])
//var meanLWold = ee.List([0.040,0.069,0.051,0.100,0.195,0.218,0.224,0.233,0.154,0.100])
//var meanSHold = ee.List([0.043,0.070,0.055,0.113,0.239,0.264,0.274,0.285,0.193,0.116])

var S2A_wavelengths = [492.4,559.8,664.6,704.1,740.5,782.8,832.8,864.7,1613.7,2202.4] 
var S2B_wavelengths = [492.1,559.0,664.9,703.8,739.1,779.7,832.9,864.0,1610.4,2185.7]

var roi = ee.FeatureCollection(test_areas.filter(ee.Filter.inList('city',[city])))
var watermask = ee.FeatureCollection(water.filter(ee.Filter.inList('city',[city])))

//print('Number of test areas in ' + city, roi.size())
//Map.addLayer(roi)

var zone = ee.Number(ee.Geometry(boundary.geometry()).centroid()
.coordinates().get(0)).divide(6).add(31).int()
var crs = ee.String('EPSG:326').cat(ee.String(zone))
//print(crs)
var proj = ee.Projection(crs)


var coefficients10 = ee.Array([
  [0.0822, 0.1360, 0.2611, 0.2964, 0.3338, 0.3877, 0.3895, 0.4750, 0.3882, 0.1366],
  [-0.1128, -0.1680, -0.3480, -0.3303, 0.0852, 0.3302, 0.3165, 0.3625, -0.4578, -0.4064],
  [0.1363, 0.2802, 0.3072, 0.5288, 0.1379, -0.0001, -0.0807, -0.1389, -0.4064, -0.5602],
]);

var sealing = function(month, image_SR, image_TOA, water_mask, end_members, roi, wl) {
  
  print(month)
  
  var waterImg = ee.Image(
    water_mask.reduceToImage({
    properties: ['type'],
    reducer: ee.Reducer.first()
  }));
  
  var image_pr = ee.Image(image_SR.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
                  .rename(['B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12'])
                  .divide(10000));
  var image_pr_toa = ee.Image(image_TOA.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
                  .rename(['B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12'])
                  .divide(10000));
                  
  var blue_mean = ee.Number((image_pr.select(['B02']).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: boundary,
    scale: 10,
    bestEffort: true})).get('B02'))
                  
  var indexes = function(image){
    
    var NDVI = image.normalizedDifference(['B08','B04']);
    
    var SAVI = image.expression (
    '(1 + L) * float((nir - red)/(nir + red + L))', {
      'nir': image.select(['B08']),
      'red': image.select(['B04']),
      'L': 0.428,
    });
    
    var NDBI = image.normalizedDifference(['B11','B08']);
    
    var MNDWI = image.normalizedDifference(['B11','B03']);
    
    var IBI_COR = image.expression(
    'float((2 * mir / (mir + nir) - ((nir / (nir + red)) \
    + (green / (green + mir)))) / (2 * mir / (mir + nir) + \
    ((nir / (nir + red)) + (green / (green + mir)))))', {
      'mir': image.select(['B11']),
      'nir': image.select(['B08']),
      'green': image.select(['B03']),
      'red': image.select(['B04']),
    });
    
    // UCI - Zhang et al., 2021 RS
    
    var UCI = image.expression(
    'float((blue - ((nir*(swir/(nir+swir)))+(swir*(nir/(nir+swir)))))/ \
    (blue + ((nir*(swir/(nir+swir)))+(swir*(nir/(nir+swir))))))', {
      'blue': image.select(['B02']),
      'nir': image.select(['B08']),
      'swir': image.select(['B11']),
    });
    
    // PISI - Tian et al., 2018 RS
  
    var PISI = image.expression(
    'float(0.8192*blue - 0.5735*nir + 0.0750)', {
      'blue': image.select(['B02']),
      'nir': image.select(['B08'])
    });
    
    var image_indexes = image_pr.addBands([NDVI.rename('NDVI'), 
                                         SAVI.rename('SAVI'), 
                                         NDBI.rename('NDBI'), 
                                         MNDWI.rename('MNDWI'),
                                         UCI.rename('UCI'),
                                         PISI.rename('PISI')]);
                                         //IBI_COR.rename('IBI_COR')]);
                                         
    var IBI_ORIG = image_indexes.expression(
    'float(((NDBI+1) - (((SAVI+1) + (MNDWI+1))/2)) / \
    ((NDBI+1) + (((SAVI+1) + (MNDWI+1))/2)))', {
      'NDBI': image_indexes.select(['NDBI']),
      'SAVI': image_indexes.select(['SAVI']),
      'MNDWI': image_indexes.select(['MNDWI']),
    });
    
    var sw12 = image_indexes.expression(
      'float(swir1/swir2)', {
        swir1: image_indexes.select(['B11']),
        swir2: image_indexes.select(['B12'])
      }).rename('SWIR_RATIO');
    
    var swir_mean = ee.Number((sw12.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: boundary,
      scale: 10,
      bestEffort: true})).get('SWIR_RATIO'))
    
    var mndwi_mean = ee.Number((image_indexes.select(['MNDWI']).pow(2).reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: boundary,
      scale: 10,
      bestEffort: true})).get('MNDWI'))
      
    var A = ee.Number(ee.Number(2).multiply(blue_mean).divide((swir_mean.add(mndwi_mean))))
    
    // Chen et al., 2019
    
    var ENDISI = image_indexes.expression(
    'float((blue - A*(swir1/swir2 + mndwipow))/(blue + A*(swir1/swir2 + mndwipow)))', {
      'blue': image_indexes.select(['B02']),
      'A': A,
      'swir1': image_indexes.select(['B11']),
      'swir2': image_indexes.select(['B12']),
      'mndwipow': image_indexes.select(['MNDWI']).pow(2),
    });
    
    return image_indexes.addBands([IBI_ORIG.rename('IBI'),ENDISI.rename('ENDISI')]);
  }
  
  var image_all = indexes(image_pr)
  
  var transform = function(image_toa) {
    
    var arrayImage1D = image_toa.toArray();
    var arrayImage2D = arrayImage1D.toArray(1);
    
    var TCTR = ee.Image(coefficients10)
    .matrixMultiply(arrayImage2D)
    .arrayProject([0])
    .arrayFlatten([['brightness', 'greenness', 'wetness']]);
    
    var getStats = function(tc) {
      
      var reducers = ee.Reducer.min().combine({
        reducer2: ee.Reducer.max(),
        sharedInputs: true
      });
      
      var stats = tc.reduceRegion({
        geometry: boundary,
        reducer: reducers,
        scale: 10,
        bestEffort: true
      })
      return tc.set(stats);
    };
    
    var TCT = getStats(TCTR);
    //print(TCT, month)
    
    var HA = TCT.expression(
      '(brightness - bmin)/(bmax - bmin)',{
        'brightness': TCT.select(['brightness']),
        'bmin': ee.Number(TCT.get('brightness_min')),
        'bmax': ee.Number(TCT.get('brightness_max')),
      });
    var VG = TCT.expression(
      '(greenness - gmin)/(gmax - gmin)',{
        'greenness': TCT.select(['greenness']),
        'gmin': ee.Number(TCT.get('greenness_min')),
        'gmax': ee.Number(TCT.get('greenness_max')),
      });
    var LA = TCT.expression(
      '(wetness - wmin)/(wmax - wmin)',{
        'wetness': TCT.select(['wetness']),
        'wmin': ee.Number(TCT.get('wetness_min')),
        'wmax': ee.Number(TCT.get('wetness_max')),
      });
    
    var TCT_calc = TCT.addBands([HA.rename('HA'),VG.rename('VG'),LA.rename('LA')]);
    
    var BCI = TCT_calc.expression(
      '((HA+LA)/2-VG)/((HA+LA)/2+VG)',{
        'HA': TCT_calc.select(['HA']),
        'LA': TCT_calc.select(['LA']),
        'VG': TCT_calc.select(['VG'])
      });
      
    var TC_BCI = TCT_calc.addBands([BCI.rename('BCI')])
    return TC_BCI
  };
  
  var transformed = transform(image_pr_toa);
  var image_all_tc = image_all.addBands(transformed);
  
  var land = image_all_tc.updateMask(waterImg.unmask(0).eq(0));
  var water_only = image_all_tc.select(['B02']).updateMask(waterImg.eq(1))
  .multiply(0).add(1).rename(['Water']);
  var unmasked = water_only.unmask(0);
  
  var options = {
    title: 'Sentinel-2 BOA spectra' + month,
    hAxis: {title: 'Wavelength (nm)'},
    vAxis: {title: 'Reflectance, %'},
    lineWidth: 1,
    pointSize: 4,
  };
  
  var SpecSign = ui.Chart.image.regions(land
  .select(['B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12']), 
  end_members, ee.Reducer.median(), 10, 'landcover', wl)
  .setChartType('ScatterChart')
  .setOptions(options);
  
  //print(SpecSign);
  
  var reflectances = land.reduceRegions({
    reducer: ee.Reducer.mean(),
    collection: em_points.filter(ee.Filter.inList('city',[city]))
    .filter(ee.Filter.inList('class',['training'])).filter('region != "NM"'),
    scale: 10,
    crs: proj
  }).map(function(item){return item.set('city', city, 'month', month)});
  
  //print(reflectances)
  
  //Export.table.toDrive({ 
  //  collection: reflectances,
  //  description: city + '_' + month + '_spectra',
  //  fileFormat: 'CSV',
  //  folder: 'Sealing_output'
  //});

  var  meanBS = land.reduceRegion(ee.Reducer.mean(), end_members
    .filter(ee.Filter.inList(em_name, ['BS'])), 10).values()
      
  var  meanDF = land.reduceRegion(ee.Reducer.mean(), end_members
    .filter(ee.Filter.inList(em_name, ['DF'])), 10).values()
      
  var  meanLW = land.reduceRegion(ee.Reducer.mean(), end_members
      .filter(ee.Filter.inList(em_name, ['LW'])), 10).values()
      
  var  meanSH = land.reduceRegion(ee.Reducer.mean(), end_members
      .filter(ee.Filter.inList(em_name, ['SH'])), 10).values()
      
  var  meanUC = land.reduceRegion(ee.Reducer.mean(), end_members
      .filter(ee.Filter.inList(em_name, ['UrbanCOL'])), 10).values()
    
  var  meanUH = land.reduceRegion(ee.Reducer.mean(), end_members
      .filter(ee.Filter.inList(em_name, ['UrbanHA'])), 10).values()
    
  var  meanUL = land.reduceRegion(ee.Reducer.mean(), end_members
      .filter(ee.Filter.inList(em_name, ['UrbanLA'])), 10).values()
    
    
  var  unm_urban = land.unmix([meanBS, meanUC, meanUH, meanUL], true, true)
  .rename(['UBSfr','UCfr','UHfr','ULfr'])
    
  var  unm_nonurban = land.unmix([meanBS, meanLW, meanSH, meanDF], true, true)
    .rename(['NBSfr','LWfr','SHfr','DFfr'])
    
  var  urban = unm_urban.expression(
        'float(UCfr + UHfr + ULfr)', {
        'UCfr': unm_urban.select(['UCfr']),
        'UHfr': unm_urban.select(['UHfr']),
        'ULfr': unm_urban.select(['ULfr']),
    }).rename(['Urban_fraction'])
    
  var  image_final = land.addBands(urban)
  
  var trans = function(feature) {
    var transformed_feature = feature.transform(proj, maxError);
  return transformed_feature;
  };
  
  var rois = ee.FeatureCollection(roi).map(trans);
  
  var mean_truth = ee.Number(rois.reduceColumns(ee.Reducer.mean(), ['seal_truth']).get('mean'))
  
  var estim = function(feature){
    
    var s_truth = ee.Number(feature.get('seal_truth'));
    
    var seal = ee.Number(feature.get('sealed'))
    .multiply(10000)
    .divide(feature.area(maxError, proj));
    
    var seal_fr = ee.Number(feature.get('Urban_fraction'))
    .multiply(10000)
    .divide(feature.area(maxError, proj));
    
    var SSR_1 = seal.subtract(s_truth);
    var SSR_2 = seal_fr.subtract(s_truth);
    
    //var SST = s_truth.subtract(mean_truth)
    
    var obs2 = ee.Number(s_truth.pow(2))
    var pred1sq = ee.Number(seal.pow(2))
    var pred2sq = ee.Number(seal_fr.pow(2))
    var mult1 = ee.Number(s_truth.multiply(seal))
    var mult2 = ee.Number(s_truth.multiply(seal_fr))
    
    return feature.set('seal', seal)
                  .set('seal_fr', seal_fr)
                  .set('SSR_1', SSR_1.pow(2))
                  .set('SSR_2', SSR_2.pow(2))
                  //.set('SST', SST.pow(2))
                  .set('OBS2', obs2)
                  .set('PRED1SQ', pred1sq)
                  .set('PRED2SQ', pred2sq)
                  .set('MULT1', mult1)
                  .set('MULT2', mult2);
  }
  
  var r2_calc = function(fc) {
    
    var n = ee.Number(fc.size())
    var obsum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['seal_truth']).get('sum'))
    var obsumsq = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['OBS2']).get('sum'))
    
    var pred1sum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['seal']).get('sum'))
    var pred2sum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['seal_fr']).get('sum'))
    
    var pred1sqsum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['PRED1SQ']).get('sum'))
    var pred2sqsum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['PRED2SQ']).get('sum'))
    
    var mult1sum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['MULT1']).get('sum'))
    var mult2sum = ee.Number(fc.reduceColumns(ee.Reducer.sum(), ['MULT2']).get('sum'))

    var r2 = ee.List([
      ee.Number(
        ee.Number(
          ee.Number(n.multiply(mult1sum)).subtract(ee.Number(obsum.multiply(pred1sum))))
          .divide(
            (ee.Number(
              ee.Number(n.multiply(obsumsq)).subtract(ee.Number(obsum).pow(2))).pow(0.5))
              .multiply(
                ee.Number(
                  ee.Number(n.multiply(pred1sqsum)).subtract(ee.Number(pred1sum).pow(2))).pow(0.5)))).pow(2),
      ee.Number(
        ee.Number(
          ee.Number(n.multiply(mult2sum)).subtract(ee.Number(obsum.multiply(pred2sum))))
          .divide(
            (ee.Number(
              ee.Number(n.multiply(obsumsq)).subtract(ee.Number(obsum).pow(2))).pow(0.5))
              .multiply(
                ee.Number(
                  ee.Number(n.multiply(pred2sqsum)).subtract(ee.Number(pred2sum).pow(2))).pow(0.5)))).pow(2)
      ]);
      
      return r2
  }
  
  var rmse_calc = function(fc) {
    
    var n = ee.Number(fc.size())
    var sumsq1 = ee.Number(fc.reduceColumns(ee.Reducer.sum(),['SSR_1']).get('sum'))
    var sumsq2 = ee.Number(fc.reduceColumns(ee.Reducer.sum(),['SSR_2']).get('sum'))

    var rmse = ee.List([
      ee.Number(sumsq1.divide(n)).pow(0.5), ee.Number(sumsq2.divide(n)).pow(0.5)
      ]);
    return rmse
  }
  
  var veg_seq = ee.List(['NDVI','SAVI']);
  var blt_seq = ee.List(['IBI','BCI','NDBI','UCI','PISI','ENDISI']);

  var veg_calc = function(element) {
    var seal_seq = function(number) {
      var value = ee.Number(number).multiply(100).round().divide(100)
      var threshold = image_final.expression(
        '((element < i ? 1 : 0) * (blue < 0.1 && swir>0.2 && nir < swir && swir2 < 0.35 ? 0 : 1)) ? 1 : 0', {
            'blue': image_final.select(['B02']),
            'nir': image_final.select(['B08']),
            'swir': image_final.select(['B11']),
            'swir2': image_final.select(['B12']),
            'element': image_final.select([element]),
            'i': value
        });
      
      var image_mask = image_final.addBands(threshold.rename(['sealed']))
      .select(['sealed']).reproject(proj,null,10);
      var urban_mask = image_final.select(['Urban_fraction'])
      .updateMask(threshold.eq(1).reproject(proj,null,10))
      var seal = image_mask.addBands(urban_mask.rename(['Urban_fraction']));
      
      var sums = seal.reduceRegions({
        collection: rois,
        reducer: ee.Reducer.sum(),
        scale: 10});
        
      var summary = sums.map(estim);
      
      var name_lst = ee.List([ee.String(element).cat(ee.String('_'))
      .cat(value),ee.String(element).cat(ee.String('fr_')).cat(value)])
      
      var r2 = r2_calc(summary);
      var rmse = rmse_calc(summary);
      
      return ee.List([name_lst, r2, rmse])
    }
    
    var summary = seq_veg.map(seal_seq);
    
    return summary
  }
  var veg_res = veg_seq.map(veg_calc)
  
  var blt_calc = function(element) {
    var seal_seq = function(number) {
    var value = ee.Number(number).multiply(100).round().divide(100)
    
    var threshold = image_final.expression(
      'element > i ? 1 : 0', {
        'element': image_final.select([element]), 
        'i': value
      });
      
    var image_mask = image_final.addBands(threshold.rename(['sealed']))
    .select(['sealed']).reproject(proj,null,10);
    var urban_mask = image_final.select(['Urban_fraction'])
    .updateMask(threshold.eq(1)).reproject(proj,null,10);
    
    var seal = image_mask.addBands(urban_mask.rename(['Urban_fraction']));
    
    var sums = seal.reduceRegions({
      collection: rois,
      reducer: ee.Reducer.sum(),
      scale: 10});
    
    var summary = sums.map(estim);
    
    var name_lst = ee.List([ee.String(element).cat(ee.String('_')).cat(value),
                            ee.String(element).cat(ee.String('fr_')).cat(value)])
      
    var r2 = r2_calc(summary);
    var rmse = rmse_calc(summary);
      
    return ee.List([name_lst, r2, rmse])
    }
    var summary2 = seq_built.map(seal_seq);
    
    return summary2
  }
  var blt_res = blt_seq.map(blt_calc)

  //var final = veg_res.cat(blt_res)
  
  var unz_names = function(list){
    return ee.List(list).unzip().get(0)
  }
  var unz_r2 = function(list){
    return ee.List(list).unzip().get(1)
  };
  
  var unz_rmse = function(list){
    return ee.List(list).unzip().get(2);
  };
  
  var final_names = ee.List(veg_res.map(unz_names).flatten()).cat(
                    ee.List(blt_res.map(unz_names).flatten()));
    
  var final_r2 = ee.List(veg_res.map(unz_r2).flatten()).cat(
    ee.List(blt_res.map(unz_r2).flatten()));
    
  var final_rmse = ee.List(veg_res.map(unz_rmse).flatten()).cat(
    ee.List(blt_res.map(unz_rmse).flatten()));
    
  var r2_max = final_r2.reduce(ee.Reducer.max());
  print('Max R2 is ', r2_max);
  
  var rmse_min = final_rmse.reduce(ee.Reducer.min());
  print('Min RMSE is ', rmse_min);
  
  var opti_par = final_names.get(ee.Array(final_r2).argmax().get(0));
  var opti_rmse = final_rmse.get(ee.Array(final_r2).argmax().get(0));
  
  print('RMSE of highest R2',opti_rmse);
  
  var index = ee.List(ee.String(opti_par).split('_')).get(0);
  var value = ee.Number.parse(ee.List(ee.String(opti_par).split('_')).get(1));
  
  //print(index);
  //print(value);
  
  //var metrics = final_r2.zip(final_rmse);
  //var metrics_names = final_names.zip(final_r2);
  
  var summary_table = ee.List.sequence(0, final_names.length().subtract(1),1).map(function(i){
    return [final_names.get(i), final_r2.get(i), final_rmse.get(i)]})
    
  var summary_fc = ee.FeatureCollection(summary_table.map(function(summary_table) {
    var index = ee.List(summary_table).get(0);
    var r2 = ee.List(summary_table).get(1);
    var rmse = ee.List(summary_table).get(2);
    
    return ee.Feature(null).set('Index', index, 'R2', r2, 'RMSE', rmse, 'Period', month, 'city', city);
  }));
  
  Export.table.toDrive({ 
    collection: summary_fc,
    selectors: ['system:index', 'Index', 'R2', 'RMSE', 'Period', 'city'],
    description: city + '_' + month + '_metrics',
    fileFormat: 'CSV',
    folder: 'Sealing_output'
  });
  
  
  //ee.Algorithms.If(veg_seq.contains(index) === true, print('Yes'), print('No'))
  
  var seal_optimal = function(image){
    
    var fr_add = function(list){
      return list.cat(
        list.map(
          function(element){
            return ee.String(element).cat('fr')}))
    };
    
    var pure_index = ee.String((ee.String(index).split('fr')).get(0));
    
    var threshold = ee.Algorithms.If(fr_add(veg_seq).contains(index),
            image.expression(
              '((index < i ? 1 : 0) * (blue < 0.1 && swir>0.2 && nir < swir && swir2 < 0.35 ? 0 : 1)) ? 1 : 0', {
                'blue': image.select(['B02']),
                'nir': image.select(['B08']),
                'swir': image.select(['B11']),
                'swir2': image.select(['B12']),
                'index': image.select([pure_index]),
                'i': value
              }),
            
            image.expression(
              'index > i ? 1 : 0', {
                'index': image.select([pure_index]),
                'i': value
              }));
    
    var sl_band = ee.Image(threshold).clip(boundary);
              
    var urban_mask = image.select(['Urban_fraction']).updateMask(sl_band.eq(1));
    
    var result = ee.Algorithms.If(ee.List(veg_seq.cat(blt_seq)).contains(index),
                    image.addBands(sl_band.rename(['sealed'])).select('sealed').updateMask(waterImg.unmask(0).eq(0)),
                    image.addBands(urban_mask.rename(['sealed'])).select(['sealed']));
                    
    return ee.Image(result);
    
  };
  
  //Map.addLayer(image_final.clip(boundary),{min:0,max:0.3,bands:['B04','B03','B02']},'True color');
  //Map.addLayer(seal_optimal(image_final).clip(boundary),{min: 0, max: 1}, 'Optimal sealed');
  
  var image_out = seal_optimal(image_final);
  
  //Export.image.toDrive({
  //  image: image_out.clip(boundary),
  //  description: city + '_' + month + '_opti_sealed', folder: 'Sealing_output',
  //  scale: 10, region: boundary, maxPixels: 200000000, crs: crs.getInfo()
  //});
};

//var test1 = sealing('April', NAK_start_SR, NAK_start_TOA, watermask, endmembers, roi, S2B_wavelengths);
//var test2 = sealing('August', NAK_mid_SR, NAK_mid_TOA, watermask, endmembers, roi, S2A_wavelengths);
//var test0 = sealing('August', ROS_mid_ext_SR, ROS_mid_ext_TOA, watermask, endmembers, roi, S2A_wavelengths);
//var test3 = sealing('November', NAK_end_SR, NAK_end_TOA, watermask, endmembers, roi, S2B_wavelengths);
//var test4 = sealing('median', NAK_median_SR, NAK_median_TOA, watermask, endmembers, roi, S2A_wavelengths);


//Export.image.toDrive({image:ROS_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'ROS_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:KUR_median_SR_NEW.select(['B2','B3','B4']).clip(boundary), 
//description:'KUR_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:KAL_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'KAL_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:PUS_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'PUS_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:MOS_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'MOS_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:CHP_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'CHP_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32637'});
//Export.image.toDrive({image:APA_median_SR.select(['B2','B3','B4']).clip(boundary), 
//description:'APA_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32636'});
//Export.image.toDrive({image:MUR_median_SR.select(['B2','B3','B4','B8','B11','B12']).clip(boundary), 
//description:'MUR_median_SR',region: boundary,scale:10, maxPixels:200000000, crs:'EPSG:32636'});


var gen_col = ee.ImageCollection([MUR_mid_SR, MUR_median_SR,
                                  APA_mid_SR, APA_median_SR,
                                  CHP_mid_SR, CHP_median_SR,
                                  MOS_mid_SR, MOS_median_SR,
                                  NSB_mid_SR, NSB_median_SR,
                                  PUS_mid_SR, PUS_median_SR,
                                  KAL_mid_SR, KAL_median_SR,
                                  KUR_mid_SR, KUR_median_SR,
                                  ROS_mid_SR, ROS_median_SR,
                                  NAK_mid_SR, NAK_median_SR]);

var sl_op = function(image_SR){
  
  var waterImg = ee.Image(
    water.reduceToImage({
    properties: ['type'],
    reducer: ee.Reducer.first()
  }));
  
  var image_pr = image_SR.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
  .rename(['B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12'])
  .divide(10000);
  
  var land = image_pr.updateMask(waterImg.unmask(0).eq(0));
    
  var NDVI = land.normalizedDifference(['B08','B04']);
  var UCI = land.expression(
    'float((blue - ((nir*(swir/(nir+swir)))+(swir*(nir/(nir+swir)))))/ \
    (blue + ((nir*(swir/(nir+swir)))+(swir*(nir/(nir+swir))))))', {
      'blue': land.select(['B02']),
      'nir': land.select(['B08']),
      'swir': land.select(['B11']),
    });
    var PISI = land.expression(
    'float(0.8192*blue - 0.5735*nir + 0.0750)', {
      'blue': land.select(['B02']),
      'nir': land.select(['B08'])
    });
    var SAVI = land.expression (
    '(1 + L) * float((nir - red)/(nir + red + L))', {
      'nir': land.select(['B08']),
      'red': land.select(['B04']),
      'L': 0.428,
    });
    var NDBI = land.normalizedDifference(['B11','B08']);
    var MNDWI = land.normalizedDifference(['B11','B03']);
    
    var image_indexes = land.addBands([NDVI.rename('NDVI'), 
                                         SAVI.rename('SAVI'), 
                                         NDBI.rename('NDBI'), 
                                         MNDWI.rename('MNDWI'),
                                         UCI.rename('UCI'),
                                         PISI.rename('PISI')]);

    var IBI_ORIG = image_indexes.expression(
    'float(((NDBI+1) - (((SAVI+1) + (MNDWI+1))/2)) / \
    ((NDBI+1) + (((SAVI+1) + (MNDWI+1))/2)))', {
      'NDBI': image_indexes.select(['NDBI']),
      'SAVI': image_indexes.select(['SAVI']),
      'MNDWI': image_indexes.select(['MNDWI']),
    });
  
  var image_all = image_indexes.addBands([IBI_ORIG.rename('IBI')]);
  
  var ndvi_th = image_all.expression(
    '((element < i ? 1 : 0) * (blue < 0.1 && swir>0.2 && nir < swir && swir2 < 0.35 ? 0 : 1)) ? 1 : 0', {
      'blue': image_all.select(['B02']),
      'nir': image_all.select(['B08']),
      'swir': image_all.select(['B11']),
      'swir2': image_all.select(['B12']),
      'element': image_all.select(['NDVI']),
      'i': 0.41
    });
  
  var pisi_th = image_all.expression(
    'element > i ? 1 : 0', {
      'element': image_all.select(['PISI']),
      'i': 0.01
    });
    
  var uci_th = image_all.expression(
    'element > i ? 1 : 0', {
      'element': image_all.select(['UCI']),
      'i': -0.49
    });
    
  var ibi_th = image_all.expression(
    'element > i ? 1 : 0', {
      'element': image_all.select(['IBI']),
      'i': -0.14
    });
    
  var image_sealed = image_all.addBands([ndvi_th.rename('NDVI_SEAL'),
                                          pisi_th.rename('PISI_SEAL'),
                                          uci_th.rename('UCI_SEAL'),
                                          ibi_th.rename('IBI_SEAL')])
                                          .select(['NDVI_SEAL','PISI_SEAL','UCI_SEAL','IBI_SEAL']);

return image_sealed

};

var sl_proc = gen_col.map(sl_op)

var worldcover = ee.Image(ee.ImageCollection("ESA/WorldCover/v100").first()).clip(cities);

var WC_reclass = function(wc){
  var SL = wc.expression(
    'Map == 50 ? 1 : 0', {
      'Map': wc.select(['Map']),
      });
      
  return(wc.addBands(SL.rename('WC_BUA'))
    .toInt()
    .select('WC_BUA')
    );
};

// WSF в GEE сильно отличается от того, что на сайте!!!!!!

//var ws_foot = ee.Image("DLR/WSF/WSF2015/v1");
//var WS_reclass = ws_foot.expression('foot == 255 ? 1 : 0', {'foot': ws_foot.select('settlement')}).rename('WS_DLR_SEAL');

var WS_reclass = WSF2019_dataset.expression('foot == 255 ? 1 : 0', {'foot': WSF2019_dataset.select('b1')}).rename('WS_DLR_SEAL');

// ESRI landcover https://gee-community-catalog.org/projects/esrilc2020/

var esri = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m").mosaic();
var ESRI_reclass = esri.expression('b1 == 7 ? 1 : 0', {'b1': esri.select('b1')}).rename('ESRI_LULC_SEAL');

var GISA = GISA_dataset.clip(cities).select('b1').rename('GISA_10m');

var proj = 'EPSG:3857';

//var batch = require('users/fitoprincipe/geetools:batch')
//batch.Download.ImageCollection.toDrive(sl_proc, 'Sealing_output', 
//                {scale: 10, 
//                 region: cities, 
//                 type: 'int'});



//var x = sl_proc.map(function(element){return element.get('system:imdex')});

//print(x)
//ee.List.sequence(0, 18, 2)

var vm = sl_proc.filter(ee.Filter.inList(
  'system:index', ee.List(
    ['0','2','4','6','8','10','12','14','16','18'])));

var med = sl_proc.filter(ee.Filter.inList(
  'system:index', ee.List(
    ['1','3','5','7','9','11','13','15','17','19'])));

var mos = med.mosaic().addBands(
  vm.mosaic()).addBands(
    WC_reclass(worldcover)).addBands(
      WS_reclass).addBands(
        ESRI_reclass).addBands(
        GISA);

var trans = function(feature) {
    var transformed_feature = feature.transform(proj, 0.001);
  return transformed_feature;
};

var sealed = mos.reduceRegions({
        collection: test_areas.map(trans),
        crs: proj,
        reducer: ee.Reducer.sum(),
        scale: 10});
        
var estim = function(feature){
  return feature.set({
    seal_UCI_med: ee.Number(feature.get('UCI_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_PISI_med: ee.Number(feature.get('PISI_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_IBI_med: ee.Number(feature.get('IBI_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_NDVI_med: ee.Number(feature.get('NDVI_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_UCI_vm: ee.Number(feature.get('UCI_SEAL_1')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_PISI_vm: ee.Number(feature.get('PISI_SEAL_1')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_IBI_vm: ee.Number(feature.get('IBI_SEAL_1')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_NDVI_vm: ee.Number(feature.get('NDVI_SEAL_1')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_ESAWC: ee.Number(feature.get('WC_BUA')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_ESRI: ee.Number(feature.get('ESRI_LULC_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_DLR: ee.Number(feature.get('WS_DLR_SEAL')).multiply(10000).divide(ee.Number(feature.area(0.001, proj))),
    seal_GISA: ee.Number(feature.get('GISA_10m')).multiply(10000).divide(ee.Number(feature.area(0.001, proj)))
  });
};
  
var updated = sealed.map(estim);

var bnd_cty = cities.filter("city == 'Moscow'")

Export.table.toDrive({ 
    collection: updated,
    description: 'summary_statistics_all_cities',
    fileFormat: 'CSV',
    folder: 'Sealing_output'
  });

Export.image.toDrive({
  image: mos.select(['WC_BUA','ESRI_LULC_SEAL','WS_DLR_SEAL','GISA_10m']).toInt16().clip(bnd_cty),
  description: 'mos_final_sealed_raster_ESA_ESRI_DLR_GISA_10m_global', folder: 'Sealing_output',
  scale: 10, region: bnd_cty, maxPixels: 700000000000, crs: proj
});


var sealed_total = mos.reduceRegions({
        collection: cities.map(trans),
        crs: proj,
        reducer: ee.Reducer.sum(),
        scale: 10});
  
var total_updated = sealed_total.map(estim);

var selectors = ['city','seal_UCI_med','seal_PISI_med','seal_IBI_med','seal_NDVI_med',
'seal_UCI_vm','seal_PISI_vm','seal_IBI_vm','seal_NDVI_vm','seal_ESAWC','seal_ESRI','seal_DLR','seal_GISA']

Export.table.toDrive({
    collection: total_updated,
    selectors: selectors,
    description: 'summary_total_sealed_all_cities',
    fileFormat: 'CSV',
    folder: 'Sealing_output'
});