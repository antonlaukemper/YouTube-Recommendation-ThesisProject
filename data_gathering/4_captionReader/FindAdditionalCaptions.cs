using System;
using System.IO;
using System.Collections;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using YoutubeExplode;
using System.Threading.Tasks;
using System.Linq;
using System.IO.Compression;

namespace captionReader
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var client = new YoutubeClient();
            // The full json with all relevant data is loaded. At this point in the pipeline it contains the information about each channel plus the id of the top three
            // videos from each channel.
            // The task of this script is to download the captions for each video
            using (StreamReader file = File.OpenText("../../output/unlabeled_data/U_channelDataWithScrapedTopVideos.json"))
            using (JsonTextReader reader = new JsonTextReader(file))
            {
                JObject channelData = (JObject)JToken.ReadFrom(reader);


                foreach(KeyValuePair<string, JToken> entry in channelData)
                {
                    // if (entry.Key == "UCtD9a-aXIYS6-e-8s7DISiw"){
                    Console.WriteLine("Now processing "+entry.Key);
                    foreach(JToken topVideo in entry.Value["top3videos"]){
                        // JToken videoId = topVideo["VideoId"];
                        JToken videoId = topVideo;   
                        string captions = null;
                        try {
                            var trackInfos = await client.GetVideoClosedCaptionTrackInfosAsync((string) videoId);
                            var trackInfo = trackInfos.First(t => t.Language.Code == "en");
                            var track = await client.GetClosedCaptionTrackAsync(trackInfo);
                            var captionList = track.Captions;
                            foreach(var line in captionList){
                                captions = captions +" "+ line;
                            }
                        } catch( System.InvalidOperationException e1 ) {
                            Console.WriteLine("no captions found for: "+ videoId);

                        } catch (YoutubeExplode.Exceptions.VideoUnavailableException e2){
                            Console.WriteLine("Video deleted: "+ videoId + " Channel: "+entry.Key);
                        } catch(System.ArgumentNullException e3){
                            Console.WriteLine("Caption could not be correctly loaded: "+ videoId);
                        }

                        var captionData = new Dictionary<string, string>()
                        {
                            { "VideoId", (string) videoId },
                            { "Captions", captions},
                            { "Info", "downloaded"}
                        };
                        var filePath = "../Data/Captions/"+entry.Key+"/"+ (string) videoId + ".jsonl";
                        if (!File.Exists(filePath+".gz")){
                            Console.WriteLine("################ "+filePath);
                            System.IO.FileInfo dirCheck = new System.IO.FileInfo(filePath);
                            dirCheck.Directory.Create(); // If the directory already exists, this method does nothing.
                            using (StreamWriter jsonFile = File.CreateText(filePath))
                            {
                                JsonSerializer serializer = new JsonSerializer();
                                //serialize object directly into file stream
                                serializer.Serialize(jsonFile, captionData);
                            }
                            var bytes = File.ReadAllBytes(filePath);
                            using (FileStream fs =new FileStream(filePath+".gz", FileMode.CreateNew))
                            using (GZipStream zipStream = new GZipStream(fs, CompressionMode.Compress, false))
                            {
                                zipStream.Write(bytes, 0, bytes.Length);
                            }
                            // Get rid of the unzipped json
                            File.Delete(filePath);
                        } else{
                            Console.WriteLine("already scraped: "+ videoId);
                        }
                    }  
                }
            }
        }
    }
}
