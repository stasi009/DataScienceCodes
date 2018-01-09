package com.wifi.news.rec;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.tensorflow.hadoop.io.TFRecordIOConf;
import org.tensorflow.hadoop.util.TFRecordWriter;

import java.io.IOException;
import java.util.Base64;

public class TFRecordOutputFormatStreaming extends FileOutputFormat<Text, Text> {
  @Override
  public RecordWriter<Text, Text> getRecordWriter(FileSystem ignored,
                                                  JobConf job, String name,
                                                  Progressable progress) throws IOException {
    Path file = FileOutputFormat.getTaskOutputPath(job, name);
    FileSystem fs = file.getFileSystem(job);

    int bufferSize = TFRecordIOConf.getBufferSize(job);
    final FSDataOutputStream fsdos = fs.create(file, true, bufferSize);
    final TFRecordWriter writer = new TFRecordWriter(fsdos);

    return new RecordWriter<Text, Text>() {
      @Override
      public void write(Text key, Text value) throws IOException {
          // key是python调用print输出的字段，为了保证准确性，在python输出时采样Base64做了encode处理；
          byte[] keyByte = Base64.getDecoder().decode(key.toString());
          writer.write(keyByte, 0, keyByte.length);
      }

      @Override
      public void close(Reporter reporter)
              throws IOException {
        fsdos.close();
      }
    };
  }

}