show collections


// EJERCICIOS PREPARADOS


// 1. Analizar con find la colección. 
db.movies.find()


// 2. Contar cuántos documentos (películas) tiene cargado. 
db.movies.find().count()


// 3. Insertar una película. 
// Primero quiero comprobar el formato de una película con los campos cast y genres rellenos
db.movies.find({
    $and: [
        { cast: { $ne: [] } },
        { genres: { $ne: [] } }
    ]
})

//Inserto mi película
db.movies.insertOne({
    "title": "Killers of the Flower Moon",
    "year": 2023,
    "cast": [
        "Leonardo DiCaprio",
        "Martin Scorsese",
        "Lily Gladstone",
        "Robert De Niro",
        "Jesse Plemons"
    ],
    "genres": [
        "Thriller",
        "Western",
        "Drama"
    ]
})

//Comprobación
db.movies.find({"title": "Killers of the Flower Moon"})


// 4. Borrar la película insertada en el punto anterior (en el 3). 
db.movies.deleteOne({ "title": "Killers of the Flower Moon" })

//Comprobación
db.movies.find({"title": "Killers of the Flower Moon"})


// 5. Contar cuantas películas tienen actores (cast) que se llaman “and”. Estos nombres de actores están por ERROR. 
db.movies.find({"cast": "and"}).count()


// 6. Actualizar los documentos cuyo actor (cast) tenga por error el valor “and” como si realmente fuera un actor. 
// Para ello, se debe sacar únicamente ese valor del array cast. Por lo tanto, no se debe eliminar ni el documento (película) ni
// su array cast con el resto de actores. 
db.movies.updateMany(
  { "cast": "and" }, // Encuentra los documentos que contienen "and" en el array "cast"
  { $pull: { "cast": "and" } } // Elimina "and" del array "cast"
)

//Comprobación
db.movies.find({"cast": "and"}).count()

 
// 7. Contar cuantos documentos (películas) tienen el array ‘cast’ vacío. 
db.movies.find({ "cast": { $eq: [] } }).count()


// 8. Actualizar TODOS los documentos (películas) que tengan el array cast vacío, añadiendo un nuevo elemento dentro
// del array con valor Undefined. Cuidado! El tipo de cast debe seguir siendo un array. El array debe ser así -> ["Undefined" ]. 
db.movies.updateMany(
  { "cast": { $eq: [] } }, // Filtro para encontrar los documentos con el array 'cast' vacío
  { $set: { "cast": ["Undefined"] } } // Actualiza el array 'cast' con el nuevo elemento
)

// Comprobación 
db.movies.find({ "cast": { $eq: [] } }).count()

// 9. Contar cuantos documentos (películas) tienen el array genres vacío. 
db.movies.find({ "genres": { $eq: [] } }).count()


// 10. Actualizar TODOS los documentos (películas) que tengan el array genres vacío, añadiendo un nuevo elemento
// dentro del array con valor Undefined. Cuidado! El tipo de genres debe seguir siendo un array. El array debe ser así -> ["Undefined" ]. 
db.movies.updateMany(
  { "genres": { $eq: [] } }, // Filtro para encontrar los documentos con el array 'cast' vacío
  { $set: { "genres": ["Undefined"] } } // Actualiza el array 'cast' con el nuevo elemento
)

// Comprobación 
db.movies.find({ "genres": { $eq: [] } }).count()

// 11. Mostrar el año más reciente / actual que tenemos sobre todas las películas. 
//FASE 1. Agrupamos todos los documentos en función del año máximo
var fase1 = {    
    $group: {
      _id: null,
      maxYear: { $max: "$year" }
    }
}
//FASE 2. Proyectamos eliminando la columna del "_id"
var fase2 = { 
    $project: {
      _id: 0,
      maxYear: 1
    }
}
db.movies.aggregate([fase1, fase2])


// 12. Contar cuántas películas han salido en los últimos 20 años. Debe hacerse desde el último año que se tienen
// registradas películas en la colección, mostrando el resultado total de esos años. Se debe hacer con el Framework de Agregación. 
//Primero convertimos el resultado obtenido en el ejercicio anterior en el valor de una variable
var fase1 = {$group: {_id: null,maxYear: { $max: "$year"}}}
var fase2 = {$project: {_id: 0,maxYear: 1}}
var yearMax = db.movies.aggregate([fase1, fase2]) //Asignamos el resultado obtenido en el ejercicio anterior a una variable nueva
var resultado = yearMax.toArray() //Transformamos el resultado obtenido en un array
var valorYearMax = resultado[0].maxYear //Nos quedamos con el valor maxYear del array

//Ahora empleamos el framework para obtener el número de películas
// FASE 1. Seleccionamos aquellas cuyo valor "year" está entre el valor obtenido previamente y ese mismo valor menos veinte
var fase1 = {
    $match: { //
      "year": { $gt: valorYearMax - 20, $lte: valorYearMax } //En este caso al usat $gt y no $gte, no estamos contando el año 1998 (ya que valorYearMax es 2018)
    }
}
// FASE 2. Agrupamos todas las películas obtenidas sumando de 1 en 1
var fase2 = { 
    $group: {
      _id: null,
      total: { $sum: 1 }
    }
}
db.movies.aggregate([fase1, fase2])


// 13. Contar cuántas películas han salido en la década de los 60 (del 60 al 69 incluidos). Se debe hacer con el Framework de Agregación. 
// FASE 1. Seleccionamos aquellas cuyo valor "year" está entre el valor 1960 y 1969
var fase1 = {
    $match: { //
      "year": { $gte: 1960, $lte: 1969 }
    }
}
// FASE 2. Agrupamos todas las películas obtenidas sumando de 1 en 1
var fase2 = { 
    $group: {
      _id: null,
      total: { $sum: 1 }
    }
}
db.movies.aggregate([fase1, fase2])


// 14. Mostrar el año u años con más películas mostrando el número de películas de ese año. 
// Revisar si varios años pueden compartir tener el mayor número de películas. 
// FASE 1. Agrupamos las películas por año
var fase1 = { 
  $group: {
    _id: {year : "$year"},
    numMovies: { $sum: 1 }
  }
}
// FASE 2. Orden descendente
var fase2 = { 
  $sort: { "numMovies": -1 }
}
// FASE 3. Nos quedamos con el #1
var fase3 = { 
  $limit: 1
}

db.movies.aggregate([fase1, fase2, fase3])

//Comprobación
db.movies.aggregate([fase1, fase2]) //No hay otros años con el mismo número de películas


// 15. Mostrar el año u años con menos películas mostrando el número de películas de ese año. 
// Revisar si varios años pueden compartir tener el menor número de películas. 
// FASE 1. Agrupamos las películas por año
var fase1 = { 
  $group: {
    _id: {year : "$year"},
    numMovies: { $sum: 1 }
  }
}
// FASE 2. Orden descendente
var fase2 = { 
  $sort: { "numMovies": 1 }
}
// FASE 3. Nos quedamos con el #1
var fase3 = { 
  $limit: 1
}

db.movies.aggregate([fase1, fase2, fase3])

//Comprobación
db.movies.aggregate([fase1, fase2]) //En este caso, 1902, 1907 y 1906 comparten el mínimo de películas/año


// 16. Guardar en nueva colección llamada “actors” realizando la fase $unwind por actor. 
// Después, contar cuantos documentos existen en la nueva colección. 
// FASE 1. Hacemos el unwind 
var fase1 = {$unwind: "$cast"}
// FASE 2. Proyectamos el resultado del unwind
var fase2 = {$project: {"_id": 0}}
//FASE . Generamos la nueva colección
var fase3 = {$out: "actors"}

db.movies.aggregate([fase1, fase2, fase3])

// Contamos el número de documentos
db.actors.find().count()

//Comprobación
db.actors.find({"title":"Caught"})
db.movies.find({"title":"Caught"})


//17. Sobre actors (nueva colección), mostrar la lista con los 5 actores que han participado en más películas mostrando el
// número de películas en las que ha participado. Importante! Se necesita previamente filtrar para descartar aquellos
// actores llamados "Undefined". Aclarar que no se eliminan de la colección, sólo que filtramos para que no aparezcan. 
//FASE 1. Filtramos descartando los actores "Undefined"
var fase1 = {
    $match: {
            cast: { $ne: "Undefined" } 
    }
}
//FASE 2. Agrupamos por el nombre del actor sumando el número de películas en las que ha participado
var fase2 = {
    $group: {
            _id: "$cast",
            numMovies: { $sum: 1 } // Contamos el número de películas por actor calculando el tamaño del array
    }
}
//FASE 3. Ordenamos por el número de películas en orden descendente
var fase3 = {$sort: { numMovies: -1 } }

//FASE 4. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.actors.aggregate([fase1, fase2, fase3, fase4])


// 18. Sobre actors (nueva colección), agrupar por película y año mostrando las 5 en las que más actores hayan
// participado, mostrando el número total de actores. 
//FASE 1. Agrupamos por película y año
var fase1 = {
    $group: {
            _id: {
                title: "$title",
                year: "$year"
            },
            cast: {$addToSet: "$cast"}
    }
}
//FASE 2. Creamos una proyección que mantenga la agrupación y que sume el total de actores en cada película
var fase2 = {
    $project: {
            _id: 1,
            totalActores: { $size: "$cast" }
    }
}
//FASE 3. Ordenamos por el número de actores en orden descendente
var fase3 = {
    $sort: { totalActores: -1 }
}

//FASE 3. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.actors.aggregate([fase1, fase2, fase3, fase4])

//Comprobación
db.movies.find({'title':"The Twilight Saga: Breaking Dawn - Part 2"}) //El número de actores coincide


// 19. Sobre actors (nueva colección), mostrar los 5 actores cuya carrera haya sido la más larga. Para ello, se debe
// mostrar cuándo comenzó su carrera, cuándo finalizó y cuántos años ha trabajado. Importante! Se necesita previamente
// filtrar para descartar aquellos actores llamados "Undefined". Aclarar que no se eliminan de la colección, sólo que filtramos para que no aparezcan. 
//FASE 1. Filtramos descartando los actores "Undefined"
var fase1 = {
    $match: {
            cast: { $ne: "Undefined" } 
    }
}
//FASE 2. Agrupamos por el nombre del actor generando campos nuevos de año máximo y mínimo
var fase2 = {
    $group: {
            _id: "$cast",
            minYear: { $min: "$year" }, 
            maxYear: { $max: "$year" } 
    }
}
//FASE 3. Generamos una proyección con un campo que calcule los años de actividad
var fase3 = {
    $project: {
        _id: 1,
        inicio: "$minYear",
        final: "$maxYear",
        actividad: {$subtract: ["$maxYear","$minYear"]} //Resta del año de la película más antigua y la más nueva
    }
}
//FASE 4. Ordenamos los actores por número de años de actividad 
var fase4 = {$sort: { actividad: -1 } }

//FASE 5. Nos quedamos con los 5 primeros
var fase5 = {$limit: 5 } 
    
db.actors.aggregate([fase1, fase2, fase3, fase4, fase5])

//Comprobación
db.actors.find({cast:"Harrison Ford"}) //Coinciden los años de inicio y final y la actividad para Harrison Ford (aunque el resultado sea irreal)
db.movies.find({cast:"Harrison Ford"})
db.movies.find({$and: [{year:2017}, {cast:"Harrison Ford"}]})


//20. Sobre actors (nueva colección), Guardar en nueva colección llamada “genres” realizando la fase $unwind por
// genres. Después, contar cuantos documentos existen en la nueva colección. 
// FASE 1. Hacemos el unwind 
var fase1 = {$unwind: "$genres"}
// FASE 2. Proyectamos el resultado del unwind
var fase2 = {$project: {"_id": 0}}
//FASE . Generamos la nueva colección
var fase3 = {$out: "genres"}

db.movies.aggregate([fase1, fase2, fase3])

// Contamos el número de documentos
db.genres.find().count()

//Comprobación
db.genres.find({"title":"Heat"})
db.movies.find({"title":"Heat"}) // Coincide


// 21. Sobre genres (nueva colección), mostrar los 5 documentos agrupados por “Año y Género” que más número de
// películas diferentes tienen mostrando el número total de películas. 
//FASE 1. Agrupamos por género y año
var fase1 = {
    $group: {
            _id: {
                genres: "$genres",
                year: "$year"
            },
            movies: {$addToSet: "$title"} //Generamos un campo películas que las vaya recopilando
    }
}
//FASE 2. Creamos una proyección que mantenga la agrupación y que sume el total de películas
var fase2 = {
    $project: {
            _id: 1,
            totalMovies: { $size: "$movies" }
    }
}
//FASE 3. Ordenamos por el número de películas en orden descendente
var fase3 = {
    $sort: { totalMovies: -1 }
}

//FASE 3. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.genres.aggregate([fase1, fase2, fase3, fase4])

//Comprobación
db.movies.find({$and: [{genres:"Drama"}, {year:1925}]}).count() //El número de películas coincide

// 22. Sobre genres (nueva colección), mostrar los 5 actores y los géneros en los que han participado con más número de
// géneros diferentes, se debe mostrar el número de géneros diferentes que ha interpretado. Importante! Se necesita
// previamente filtrar para descartar aquellos actores llamados "Undefined". Aclarar que no se eliminan de la colección,
// sólo que filtramos para que no aparezcan. 
//FASE 1. Hacemos un unwind para poder relacionar cada actor con sus géneros correspondientes
var fase1 = {
    $unwind: "$cast"
}
//FASE 2. Filtramos descartando los actores "Undefined"
var fase2 = {
    $match: {
            cast: { $ne: "Undefined" }, // Habría que usar un $elemMatch para ir comprobando cada elemento del array?
            //genres: { $ne: "Undefined" } // Podríamos también descartar el género "Undefined"
    }
}
//FASE 3. Agrupamos por el nombre del actor generando un campo que añada los géneros contenidos en el array genres
var fase3 = {
    $group: {
            _id: "$cast",
            genres: { $addToSet: "$genres" }, //Usamos $addToSet para evitar duplicidades
    }
}
//FASE 4. Proyectamos generando un nuevo campo que cuente el número de elementos del array genres
var fase4 = {
    $project: {
        _id : 1, 
        genres : 1,
        count : {$size: "$genres"} //Obtenemos en esta columna el número de géneros diferentes
    }
}
//FASE 5. Ordenamos por número de géneros diferentes
var fase5 = {$sort: {count: -1 } }

//FASE 6. Nos quedamos con los 5 primeros
var fase6 = {$limit: 5 } 
    
db.genres.aggregate([fase1, fase2, fase3, fase4, fase5, fase6])

//Comprobación
db.genres.distinct("genres", { cast: "Johnny Depp" }).count() //Coincide


// 23. Sobre genres (nueva colección), mostrar las 5 películas y su año correspondiente en los que más géneros diferentes
// han sido catalogados, mostrando esos géneros y el número de géneros que contiene. 
//FASE 1. Agrupamos por el título y año de cada película agregando los genres a un único campo (estamos "deshaciendo" el unwind)
var fase1 = {
    $group: {
            _id: {
                title : "$title", 
                year : "$year"
                
            },
            genres: { $addToSet: "$genres" }, //Usamos $addToSet para evitar duplicidades
    }
}
//FASE 2. Proyectamos generando un nuevo campo que cuente el número de elementos del array genres
var fase2 = {
    $project: {
        _id : 1, 
        genres : 1,
        count : {$size: "$genres"} 
    }
}
//FASE 3. Ordenamos por número de géneros diferentes
var fase3 = {$sort: {count: -1 } }

//FASE 4. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.genres.aggregate([fase1, fase2, fase3, fase4])

//Comprobación
db.genres.find({title:"American Made"})


// EJERCICIOS LIBRES


// 1. Top 5 actores más activos, es decir, que tengan mejor relación películas rodadas/años de carrera
//FASE 1. Eliminamos actores "Undefined"
var fase1 = {
    $match: {
            cast: { $ne: "Undefined" } 
    }
}
//FASE 2. Agrupamos por el nombre del actor generando campos nuevos de año máximo y mínimo y contamos el número de películas en su haber
var fase2 = {
    $group: {
            _id: "$cast",
            minYear: { $min: "$year" }, 
            maxYear: { $max: "$year" }, 
            count: {$sum:1}
    }
}
//FASE 3. Generamos una proyección con un campo que calcule los años de actividad y la relación
var fase3 = {
    $project: {
        _id: 1,
        inicio: "$minYear",
        final: "$maxYear",
        actividad: { $add: [{ $subtract: ["$maxYear", "$minYear"] }, 1] }, //Resta del año de la película más antigua y la más nueva y le suma 1 (para aquellos actores que solo han rodado en un año)
        numMovies: "$count",
        relNumYears: { //Calculamos la relación dividiendo
        $divide: [
            "$count",
            { $add: [{ $subtract: ["$maxYear", "$minYear"] }, 1] }
    ]
}

    }
}
//FASE 4. Ordenamos los actores por número de años de actividad 
var fase4 = {$sort: { relNumYears: -1 } }

//FASE 5. Nos quedamos con los 5 primeros
var fase5 = {$limit: 5 } 
    
db.actors.aggregate([fase1, fase2, fase3, fase4, fase5])

//Comprobación
db.movies.find({cast:"William Garwood"}) //Coincide


// 2. Top 5 géneros menos populares de la historia, es decir, presentes en menor número de años
//FASE 1. Agrupamos por año generando un nuevo campo que reúna los actores activos cada año
var fase1 = {
    $group: {
            _id: "$genres",
            yearsON: { $addToSet: "$year" }, //Usamos $addToSet para evitar duplicidades
    }
}
//FASE 2. Proyectamos generando un nuevo campo que cuente el número de elementos del array actors
var fase2 = {
    $project: {
        _id : 1, 
        count : {$size: "$yearsON"} 
    }
}
//FASE 3. Ordenamos por número de géneros diferentes
var fase3 = {$sort: {count: 1 } }

//FASE 4. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.genres.aggregate([fase1, fase2, fase3, fase4])

//Comprobación
db.genres.distinct("year",{genres:"Supernatural"}).length //Coincide


// 3. Año con mayor participación (más actores activos)
//FASE 1. Agrupamos por año generando un nuevo campo que reúna los actores activos cada año
var fase1 = {
    $group: {
            _id: "$year",
            actors: { $addToSet: "$cast" }, //Usamos $addToSet para evitar duplicidades
    }
}
//FASE 2. Proyectamos generando un nuevo campo que cuente el número de elementos del array actors
var fase2 = {
    $project: {
        _id : 1, 
        count : {$size: "$actors"} 
    }
}
//FASE 3. Ordenamos por número de géneros diferentes
var fase3 = {$sort: {count: -1 } }

//FASE 4. Nos quedamos con los 5 primeros
var fase4 = {$limit: 5 } 
    
db.actors.aggregate([fase1, fase2, fase3, fase4])

//Comprobación
db.actors.distinct("cast",{year:2012}) //Coincide

